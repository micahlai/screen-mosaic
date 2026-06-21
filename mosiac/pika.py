"""
Pika API wrapper.

Set the PIKA_API_KEY environment variable before calling generate().
Get your key from https://pika.art/
"""

import os
import time
import requests

PIKA_API_BASE = "https://api.pika.art/v1"


def generate(prompt: str, duration: int = 5, aspect_ratio: str = "16:9",
             loop: bool = True) -> bytes:
    """
    Generate a video with Pika and return the raw mp4 bytes.
    Blocks until the video is ready (polls every 5 seconds).

    Args:
        prompt:       Text description of the video to generate.
        duration:     Length in seconds (3 or 5).
        aspect_ratio: "16:9" | "9:16" | "1:1"
        loop:         Whether to generate a seamlessly looping video.

    Returns:
        Raw mp4 bytes.

    Raises:
        RuntimeError: if the API key is missing, the request fails, or
                      the generation times out after 3 minutes.
    """
    api_key = os.environ.get("PIKA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "PIKA_API_KEY environment variable is not set. "
            "Get your key at https://pika.art/ and run:\n"
            "  export PIKA_API_KEY=your_key_here"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Submit generation request
    payload = {
        "prompt": prompt,
        "options": {
            "duration": duration,
            "aspectRatio": aspect_ratio,
            "loop": loop,
        },
    }
    resp = requests.post(f"{PIKA_API_BASE}/generate", json=payload, headers=headers)
    if not resp.ok:
        raise RuntimeError(f"Pika API error {resp.status_code}: {resp.text}")

    job = resp.json()
    job_id = job.get("id") or job.get("jobId") or job.get("job_id")
    if not job_id:
        raise RuntimeError(f"No job ID in Pika response: {job}")

    print(f"[pika] Job submitted: {job_id}. Waiting for video...")

    # Poll until done (up to 3 minutes)
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(5)
        poll = requests.get(f"{PIKA_API_BASE}/jobs/{job_id}", headers=headers)
        if not poll.ok:
            raise RuntimeError(f"Pika poll error {poll.status_code}: {poll.text}")
        data = poll.json()
        state = data.get("status") or data.get("state", "")
        print(f"[pika] Status: {state}")

        if state in ("completed", "succeeded", "finished", "done"):
            video_url = (data.get("video", {}).get("url")
                         or data.get("resultUrl")
                         or data.get("url"))
            if not video_url:
                raise RuntimeError(f"No video URL in completed job: {data}")
            print(f"[pika] Downloading video from {video_url}")
            video_resp = requests.get(video_url)
            video_resp.raise_for_status()
            return video_resp.content

        if state in ("failed", "error"):
            raise RuntimeError(f"Pika generation failed: {data}")

    raise RuntimeError("Pika generation timed out after 3 minutes.")
