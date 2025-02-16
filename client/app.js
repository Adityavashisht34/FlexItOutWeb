const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const repCount = document.getElementById('rep-count');
const statusText = document.getElementById('status');

let currentExercise = null;
let repCounter = 0;
let isProcessing = false;

// Get camera access
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve();
            };
        });
    } catch (error) {
        console.error('Error accessing camera:', error);
        statusText.textContent = 'Error accessing camera. Please allow camera permissions.';
    }
}

// Handle exercise selection
document.querySelectorAll('.exercise-selector button').forEach(button => {
    button.addEventListener('click', () => {
        currentExercise = button.textContent;
        statusText.textContent = `Tracking ${currentExercise}...`;
        repCounter = 0;
        repCount.textContent = '0';
        startProcessing();
    });
});

// Process frames and send to server
async function startProcessing() {
    if (!currentExercise || isProcessing) return;
    
    isProcessing = true;
    const context = canvas.getContext('2d');
    
    async function processFrame() {
        if (!currentExercise) {
            isProcessing = false;
            return;
        }

        // Capture frame from video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert frame to base64
        const frame = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        
        try {
            const response = await fetch('https://FlexItOutServer.adityatwitter12.repl.co/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    frame: frame,
                    exercise_type: currentExercise
                })
            });
            
            const data = await response.json();
            if (data.reps > 0) {
                repCounter += data.reps;
                repCount.textContent = repCounter;
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        }
        
        requestAnimationFrame(processFrame);
    }
    
    processFrame();
}

// Initialize the app
setupCamera();
