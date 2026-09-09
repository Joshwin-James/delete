import { HandLandmarker, ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const statusText = document.getElementById("status");
const captureBtn = document.getElementById("captureBtn");
const startBtn = document.getElementById("startBtn");
const drawBboxCheckbox = document.getElementById("drawBbox");
const countdownText = document.getElementById("countdown");

let handLandmarker;
let objectDetector;
let backgroundCanvas = document.createElement("canvas");
let backgroundCtx = backgroundCanvas.getContext("2d");

let hasBackground = false;
let isAppRunning = false;
let lastVideoTime = -1;

// Helper: Calculate distance between two points
function dist(p1, p2) {
    return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

// Logic: Check if the hand is closed (less than 3 fingers extended)
function isHandClosed(landmarks) {
    const wrist = landmarks[0];
    const fingerPairs = [[8, 6], [12, 10], [16, 14], [20, 18]]; // tip vs pip
    let extendedFingers = 0;
    
    for (const [tip, pip] of fingerPairs) {
        if (dist(landmarks[tip], wrist) > dist(landmarks[pip], wrist)) {
            extendedFingers++;
        }
    }
    return extendedFingers < 3;
}

// Debounce logic (KeepState equivalent)
let keepFor = 6; // frames to debounce
let debounceCount = 6;
let currentState = false;
function updateState(handClosed) {
    if (handClosed === currentState) {
        debounceCount++;
        if (debounceCount > keepFor) debounceCount = keepFor;
        return currentState;
    }
    debounceCount--;
    if (debounceCount < 0) {
        debounceCount = keepFor;
        currentState = handClosed;
    }
    return currentState;
}

// Initialize MediaPipe and Webcam
async function init() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        statusText.innerText = "Loading Hand Landmarker...";
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 1
        });

        statusText.innerText = "Loading Object Detector...";
        objectDetector = await ObjectDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "efficientdet_lite0.tflite",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            scoreThreshold: 0.5,
            categoryAllowlist: ["person"]
        });

        statusText.innerText = "Requesting webcam access...";
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        
        captureBtn.disabled = false;
        statusText.innerText = "Ready! Step 1: Clear the scene and click Capture Background.";
    } catch (err) {
        console.error(err);
        statusText.innerText = "Error: " + err.message;
        statusText.style.color = "red";
    }
}

// Main Render Loop
async function predictWebcam() {
    // Set internal canvas dimensions to match video stream
    if (canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
        backgroundCanvas.width = video.videoWidth;
        backgroundCanvas.height = video.videoHeight;
    }

    let startTimeMs = performance.now();
    
    // Only process if we have a new frame
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        
        if (isAppRunning && hasBackground) {
            let handClosed = false;
            
            // 1. Detect Hand
            const handResults = handLandmarker.detectForVideo(video, startTimeMs);
            if (handResults.landmarks.length > 0) {
                handClosed = isHandClosed(handResults.landmarks[0]);
            }
            
            // Debounce the state
            let activeState = updateState(handClosed);

            // 2. If hand is closed, find person and erase
            if (activeState) {
                const objResults = objectDetector.detectForVideo(video, startTimeMs);
                let personBox = null;
                let maxArea = 0;
                
                // Find largest person
                for (let det of objResults.detections) {
                    let b = det.boundingBox;
                    let area = b.width * b.height;
                    if (area > maxArea) {
                        maxArea = area;
                        let pad = 10;
                        let x = Math.max(0, b.originX - pad);
                        let y = Math.max(0, b.originY - pad);
                        let w = Math.min(canvasElement.width - x, b.width + pad * 2);
                        let h = canvasElement.height - y;
                        personBox = {x, y, w, h};
                    }
                }

                // Draw current video frame first
                canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

                if (personBox) {
                    // Overwrite the person's bounding box with the captured background image
                    canvasCtx.drawImage(
                        backgroundCanvas,
                        personBox.x, personBox.y, personBox.w, personBox.h, // Source (background)
                        personBox.x, personBox.y, personBox.w, personBox.h  // Destination (canvas)
                    );
                    
                    // Draw bounding box if enabled
                    if (drawBboxCheckbox.checked) {
                        canvasCtx.strokeStyle = "#00FF00";
                        canvasCtx.lineWidth = 2;
                        canvasCtx.strokeRect(personBox.x, personBox.y, personBox.w, personBox.h);
                    }
                }
            } else {
                // Hand open or not detected -> Just draw the video
                canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
            }
        } else {
            // App not running -> Just draw the video
            canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
        }
    }
    
    window.requestAnimationFrame(predictWebcam);
}

// Background Capture Logic
captureBtn.addEventListener("click", () => {
    captureBtn.disabled = true;
    startBtn.disabled = true;
    
    let count = 5;
    countdownText.classList.remove("hidden");
    statusText.innerText = "Move out of the frame!";
    
    let timer = setInterval(() => {
        countdownText.innerText = count > 0 ? count : "Snap!";
        if (count === 0) {
            clearInterval(timer);
            // Save current video frame to off-screen background canvas
            backgroundCtx.drawImage(video, 0, 0, backgroundCanvas.width, backgroundCanvas.height);
            hasBackground = true;
            
            setTimeout(() => {
                countdownText.classList.add("hidden");
                captureBtn.disabled = false;
                startBtn.disabled = false;
                statusText.innerText = "Background captured! Click Start App when ready.";
            }, 500);
        }
        count--;
    }, 1000);
});

// App Toggle Logic
startBtn.addEventListener("click", () => {
    isAppRunning = !isAppRunning;
    startBtn.innerText = isAppRunning ? "Stop App" : "Start App";
    startBtn.style.background = isAppRunning ? "#dc3545" : "#007bff";
    statusText.innerText = isAppRunning ? "App running! Close your fist to delete yourself." : "App stopped.";
});

// Start the app!
init();
