I need help Dockerizing my distributed camera streaming + middleware processing setup for a smart manufacturing vision pipeline.

Current architecture:
- I have 2 Raspberry Pis, each with 1 USB global-shutter camera attached at /dev/video0.
- Each Pi currently runs an ffmpeg command that reads from /dev/video0 using v4l2 and streams RTP/H.264 over UDP to a middleware machine.
- The middleware machine runs a Python project called camera_sensor_fusion that receives both RTP streams and processes them.
- Current mapping:
  - cam1 Pi IP = 192.168.1.80, streams to middleware 192.168.1.113:5000
  - cam2 Pi IP = 192.168.1.92, streams to middleware 192.168.1.113:5001
- Current middleware command:
  cd ~/camera_sensor_fusion
  source .venv-system/bin/activate
  python -m camera_fusion.launch configs/cam1.json configs/cam2.json
- Current Pi ffmpeg command pattern:
  ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 \
  -i /dev/video0 \
  -an \
  -c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency \
  -payload_type 96 \
  -f rtp rtp://192.168.1.113:5000
  and same for cam2 except port 5001.

Goal:
I want to containerize this setup cleanly so that:
1. each camera Pi runs one reusable Docker container for camera streaming
2. the middleware machine runs one Docker container for the Python receiver/processor
3. camera-specific settings are configurable without editing source code
4. the setup scales easily when I add more Pis/cameras later

Please generate the files and code needed for this setup.

What I want you to create:
1. A reusable Pi camera-stream container
   - Dockerfile for the Pi stream service
   - docker-compose.yml for the Pi stream service
   - entrypoint shell script (or similar) that builds and runs the ffmpeg command
   - use environment variables for:
     - CAMERA_NAME
     - VIDEO_DEVICE
     - DEST_IP
     - DEST_PORT
     - WIDTH
     - HEIGHT
     - FPS
   - map /dev/video0 into the container
   - use host networking if appropriate for RTP/UDP simplicity
   - make the container restart unless stopped
   - keep the image lightweight if possible

2. A middleware container
   - Dockerfile for the middleware Python service
   - docker-compose.yml for the middleware service
   - run:
     python -m camera_fusion.launch configs/cam1.json configs/cam2.json
   - assume the project root is camera_sensor_fusion
   - install all needed Python dependencies
   - if OpenCV/GStreamer system dependencies are needed, include them
   - mount configs/calibration/data folders as volumes so outputs persist outside the container
   - use host networking if appropriate for receiving RTP on ports 5000 and 5001
   - restart unless stopped

3. Project structure recommendations
   - suggest a clean folder layout for:
     - pi-stream/
     - middleware/
     - scripts/
     - configs/
     - data/
   - keep it understandable for a student research lab project

4. Example environment files
   - .env example for cam1 Pi
   - .env example for cam2 Pi
   - clearly show how only DEST_PORT / CAMERA_NAME or other variables change

5. Exact run instructions
   - how to build and start the middleware container
   - how to build and start each Pi stream container
   - how to stop them
   - how to inspect logs

6. Notes about likely pitfalls
   - /dev/video0 permissions
   - host networking on Linux
   - UDP/RTP port handling
   - volume mounts
   - ffmpeg availability
   - OpenCV/GStreamer issues in containers

Important constraints:
- Do not redesign the whole architecture. Keep the same architecture I already use:
  - Pi opens USB camera
  - Pi sends RTP/H.264 to middleware
  - middleware receives both streams and runs camera_sensor_fusion
- Make the solution practical and minimal, not overengineered
- Prefer docker compose over raw docker run commands
- Use clear comments in every file
- Make the generated scripts production-like but still easy for me to understand and edit
- If assumptions are needed, state them clearly in comments

Please output:
- each file separately with filenames
- full contents for each file
- then a short explanation of how everything works together
- then the exact commands I should run on middleware, cam1 Pi, and cam2 Pi