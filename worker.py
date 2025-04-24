import os, io, json, zipfile, shutil, time, requests
from datetime import datetime
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

INCOMING = "comphy_jobs/incoming"
RESULTS = "comphy_jobs/results"
OUTPUT_FOLDER = "./ComfyUI/output/"
COMPHY_API_URL = "http://localhost:8188"
POLL_INTERVAL = 15  # seconds
WAIT_SECONDS = 5 # wait for a job to finish
TEMP_DIR = "tmp_comphy_worker"
RUN_JSON_NAME = "run.json"
SCOPES = ['https://www.googleapis.com/auth/drive']


def get_drive_service():

    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    import pickle

    creds = None
    token_path = 'token.pickle'
    credentials_path = 'credentials.json'

    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

def find_folder(service, folder_name, parent_id=None):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return results.get("files", [])[0] if results.get("files") else None

def walk_path(service, path):
    parts = path.split("/")
    parent = None
    for part in parts:
        folder = find_folder(service, part, parent)
        if not folder:
            raise Exception(f"Folder '{part}' not found")
        parent = folder["id"]
    return parent

def get_output_file_set():
    file_set = set()
    for root, _, files in os.walk(OUTPUT_FOLDER):
        for f in files:
            file_set.add(os.path.join(root, f))
    return file_set

def list_zip_jobs(service, folder_id):
    query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and name contains '.zip'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return results.get("files", [])

def download_zip(service, file_id, dest_path):
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

def upload_results_folder(service, output_files, drive_base_folder_name=RESULTS):
    def get_or_create_drive_folder(service, parent_id, folder_name):
        # Check if folder exists
        query = f"'{parent_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = service.files().list(q=query, spaces='drive', fields="files(id, name)").execute()
        items = results.get("files", [])
        if items:
            return items[0]["id"]

        # Create it
        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id]
        }
        file = service.files().create(body=file_metadata, fields="id").execute()
        return file["id"]

    def get_or_create_nested_folders(service, path_parts):
        parent_id = "root"
        for part in path_parts:
            parent_id = get_or_create_drive_folder(service, parent_id, part)
        return parent_id

    uploaded = []

    for file_path in output_files:
        if not os.path.isfile(file_path):
            continue

        rel_path = os.path.relpath(file_path, start=TEMP_DIR)
        path_parts = os.path.normpath(os.path.join(drive_base_folder_name, rel_path)).split(os.sep)
        filename = path_parts.pop()  # Extract file name
        drive_folder_id = get_or_create_nested_folders(service, path_parts)

        media = MediaFileUpload(file_path, resumable=True)
        file_metadata = {
            "name": filename,
            "parents": [drive_folder_id]
        }

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id"
        ).execute()

        uploaded.append(file_path)
        print(f"✅ Uploaded {file_path} to {drive_base_folder_name}")

    return uploaded


def copy_resource_files(tempdir):
    info_path = os.path.join(tempdir, "info.json")
    copied_files = []

    if not os.path.exists(info_path):
        print("⚠️ No info.json found — skipping resource copy.")
        return copied_files

    with open(info_path, "r") as f:
        info = json.load(f)

    audio_dst = info.get("audio_path")
    image_dst = info.get("image_path")

    audio_src = os.path.join(tempdir, "audio_file")
    image_src = os.path.join(tempdir, "image_file")

    if audio_dst and not os.path.exists(audio_dst) and os.path.exists(audio_src):
        os.makedirs(os.path.dirname(audio_dst), exist_ok=True)
        shutil.copyfile(audio_src, audio_dst)
        copied_files.append(audio_dst)
    else:
        print(f"⚠️ Did not copy {audio_dst}")

    if image_dst and not os.path.exists(image_dst) and os.path.exists(image_src):
        os.makedirs(os.path.dirname(image_dst), exist_ok=True)
        shutil.copyfile(image_src, image_dst)
        copied_files.append(image_dst)
    else:
        print(f"⚠️ Did not copy {image_dst}")

    return copied_files


def delete_resource_files(file_list):
    for file_path in file_list:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🗑️ Deleted: {file_path}")
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")


def check_comphy():
    try:
        response = requests.get(f"{COMPHY_API_URL}/queue", timeout=2)
        if response.status_code == 200:
            queue = response.json()
            queue_len = len(queue["queue_running"]) + len(queue["queue_pending"])
            return queue_len == 0
        else:
            print(f"🟠 ComphyUI responded with unexpected status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"🔴 ComphyUI not reachable: {e}")
        return False

def wait_for_queue_empty():
    while True:
        try:
            response = requests.get(f"{COMPHY_API_URL}/queue")
            queue = response.json()
            queue_len = len(queue["queue_running"]) + len(queue["queue_pending"])
            if queue_len == 0:
                break
        except Exception as e:
            print(f"Queue check failed: {e}")
        time.sleep(WAIT_SECONDS)


def run_comphy_job(run_json_path, temp_output_dir):
    with open(run_json_path, "r") as f:
        workflow_data = json.load(f)

    if "prompt" not in workflow_data:
        workflow_data = { "prompt": workflow_data }

    # Ensure temp output dir exists
    os.makedirs(temp_output_dir, exist_ok=True)

    # 1. Snapshot before
    before_files = get_output_file_set()

    # 2. Submit job
    print("🚀 Submitting job to ComphyUI")
    response = requests.post(f"{COMPHY_API_URL}/prompt", json=workflow_data)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to queue job: {response.status_code} {response.text}")

    # 3. Wait for job to complete
    print("⏳ Waiting for queue to finish...")
    wait_for_queue_empty()

    # 4. Snapshot after
    after_files = get_output_file_set()

    # 5. Determine new files
    new_files = sorted(list(after_files - before_files))
    moved_files = []

    # 6. Move new files to TEMP_DIR
    for f in new_files:
        rel_path = os.path.relpath(f, OUTPUT_FOLDER)
        target_path = os.path.join(temp_output_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.move(f, target_path)
        moved_files.append(target_path)

    print(f"✅ Job completed — {len(moved_files)} new files moved to {temp_output_dir}")
    return moved_files


def main():
    service = get_drive_service()
    incoming_id = walk_path(service, INCOMING)
    results_id = walk_path(service, RESULTS)

    os.makedirs(TEMP_DIR, exist_ok=True)

    print("✅ Comphy Drive Worker started.")
    while True:

        # check if comphy is running
        if not check_comphy():
            time.sleep(POLL_INTERVAL)
            continue            

        # if there is already a run.json in the workfolder
        if os.path.exists(os.path.join(TEMP_DIR, RUN_JSON_NAME)):
            print("⚠️ Found old job. Retry it!")
        else:
            # check cloud for new jobs
            jobs = list_zip_jobs(service, incoming_id)
            if not jobs:
                time.sleep(POLL_INTERVAL)
                continue

            job = jobs[0]  # process one at a time
            job_name = job["name"]
            job_id = job["id"]
            local_zip = os.path.join(TEMP_DIR, job_name)

            print(f"⬇️  Downloading job: {job_name}")
            download_zip(service, job_id, local_zip)

            # Delete .zip from drive
            service.files().delete(fileId=job_id).execute()

            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(TEMP_DIR)

        run_json_path = os.path.join(TEMP_DIR, RUN_JSON_NAME)
        if not os.path.exists(run_json_path):
            print("⚠️ No run.json found — skipping.")
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)
            continue

        # copy the resources
        copied_files = copy_resource_files(TEMP_DIR)
        
        # run the process
        output_files = run_comphy_job(os.path.join(TEMP_DIR, "run.json"), TEMP_DIR)


        # delete the copied resources
        delete_resource_files(copied_files)

        print("⬆️  Uploading results")
        upload_results_folder(service, output_files)

        print("🧹 Cleaning up...")
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)



        print("✅ Job complete.\n")

if __name__ == "__main__":
    main()