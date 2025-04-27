// CarbonSense Voice Handler JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('[Voice] Initializing voice recording module');
    setupVoiceRecording();
});

// Setup voice recording functionality
function setupVoiceRecording() {
    const micButton = document.getElementById('mic-button');
    const stopRecordingButton = document.getElementById('stop-recording-button');
    const recordingIndicator = document.getElementById('recording-indicator');
    const recordingTimeElement = document.getElementById('recording-time');
    const conversationContainer = document.getElementById('conversation-container');
    
    // Exit if elements don't exist
    if (!micButton || !stopRecordingButton || !recordingIndicator) {
        console.warn('[Voice] Voice recording elements not found in the DOM');
        return;
    }
    
    console.log('[Voice] Voice recording components initialized');
    
    let sessionId = null;
    let recordingTimer = null;
    let recordingSeconds = 0;
    let resultCheckTimer = null;
    let isRecording = false;
    let recordingStopped = false;
    let manuallyStoppedRecording = false;
    
    // Start recording when mic button is clicked
    micButton.addEventListener('click', function() {
        console.log('[Voice] Mic button clicked');
        if (isRecording) {
            console.log('[Voice] Already recording, ignoring click');
            return; // Prevent multiple recordings
        }
        
        startRecording();
    });
    
    // Stop recording when stop button is clicked
    stopRecordingButton.addEventListener('click', function() {
        console.log('[Voice] Stop button clicked');
        manuallyStoppedRecording = true;
        stopRecording();
    });
    
    // ----- STEP 1: RECORDING -----
    // Start the recording process immediately
    function startRecording() {
        console.log('[Voice] Starting voice recording');
        isRecording = true;
        recordingStopped = false;
        manuallyStoppedRecording = false;
        
        // Show recording indicator and hide mic button
        recordingIndicator.style.display = 'flex';
        micButton.classList.add('recording');
        console.log('[Voice] Recording UI updated - indicator visible');
        
        // Reset recording time
        recordingSeconds = 0;
        recordingTimeElement.textContent = recordingSeconds;
        
        // Add a "recording" message to the chat
        showStatusIndicator('recording');
        
        // Start timer to update recording time display
        recordingTimer = setInterval(function() {
            recordingSeconds++;
            recordingTimeElement.textContent = recordingSeconds;
            
            // Auto-stop after 15 seconds (matching backend timeout) to prevent very long recordings
            if (recordingSeconds >= 15) {
                console.log('[Voice] Recording time limit reached (15s). Auto-stopping');
                stopRecording();
            }
        }, 1000);
        console.log('[Voice] Recording timer started');
        
        // Send request to start recording on the server
        console.log('[Voice] Sending start-recording API request');
        fetch('/api/start-recording', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            console.log('[Voice] Start recording response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('[Voice] Start recording API response:', data);
            if (data.status === 'recording_started') {
                sessionId = data.session_id;
                console.log('[Voice] Recording session started with ID:', sessionId);
            } else {
                console.error('[Voice] Failed to start recording:', data);
                stopRecording(true);
                showErrorMessage('Failed to start voice recording. Please try again.');
            }
        })
        .catch(error => {
            console.error('[Voice] Error starting recording:', error);
            stopRecording(true);
            showErrorMessage('Network error when starting voice recording.');
        });
    }
    
    // ----- STEP 2: TRANSCRIBING -----
    // Stop the recording process and move to transcription phase
    function stopRecording(isError = false) {
        if (recordingStopped) {
            console.log('[Voice] Recording already stopped, ignoring duplicate stop');
            return; // Prevent multiple stop calls
        }
        
        recordingStopped = true;
        console.log('[Voice] Stopping voice recording, isError =', isError);
        
        // Clear timers
        if (recordingTimer) {
            clearInterval(recordingTimer);
            recordingTimer = null;
            console.log('[Voice] Recording timer cleared');
        }
        
        // Reset UI state
        recordingIndicator.style.display = 'none';
        micButton.classList.remove('recording');
        isRecording = false;
        console.log('[Voice] Recording UI reset - indicator hidden');
        
        // If there was an error, show message and exit
        if (isError) {
            console.log('[Voice] Error occurred, removing status indicator');
            removeStatusIndicator();
            return;
        }
        
        // If we have a session, proceed to transcription
        if (sessionId) {
            // Update UI to show transcribing phase
            showStatusIndicator('transcribing');
            console.log('[Voice] Showing transcription in progress');
            
            // Call API to stop recording
            console.log('[Voice] Sending stop-recording API request for session:', sessionId);
            fetch(`/api/stop-recording/${sessionId}`, {
                method: 'POST'
            })
            .then(response => {
                console.log('[Voice] Stop recording response status:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('[Voice] Stop recording API response:', data);
                
                // ----- STEP 3: FETCH TRANSCRIPTION AFTER DELAY -----
                // Wait a short delay before starting to check for the transcript
                console.log('[Voice] Waiting 3 seconds before starting transcript checks');
                setTimeout(fetchTranscription, 3000);
            })
            .catch(error => {
                console.error('[Voice] Error stopping recording:', error);
                showErrorMessage('Error processing your voice. Please try again.');
                removeStatusIndicator();
            });
        }
    }
    
    // Fetch transcription after delay
    function fetchTranscription() {
        console.log('[Voice] Fetching transcription after delay');
        
        // Implement polling for the transcript file
        let checkCount = 0;
        const maxChecks = 10; // Maximum number of checks
        const checkInterval = 1500; // Check every 1.5 seconds
        let transcriptFound = false;  // Flag to track if transcript was found
        
        function checkTranscriptFile() {
            // If transcript already found, don't check again
            if (transcriptFound) {
                console.log('[Voice] Transcript already found, skipping check');
                return;
            }
            
            checkCount++;
            console.log(`[Voice] Checking for transcript (attempt ${checkCount}/${maxChecks})`);
            
            fetch('/api/check-transcript-file', {
                method: 'GET'
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('[Voice] Transcript file check result:', data);
                
                if (data.status === 'success' && data.transcript) {
                    // We have a transcript, display it
                    transcriptFound = true;  // Set flag to prevent further checks
                    displayTranscription(data.transcript);
                    
                    // Start the result generation phase
                    startResultGeneration();
                } else if (checkCount < maxChecks) {
                    // Schedule another check
                    setTimeout(checkTranscriptFile, checkInterval);
                } else {
                    // Fallback to the original check-processing endpoint
                    console.log('[Voice] Falling back to check-processing endpoint');
                    
                    fetch(`/api/check-processing/${sessionId}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('[Voice] Transcription check result:', data);
                        
                        // If we have a transcription, display it
                        if (data.status === 'complete' && data.transcription) {
                            // Show transcription and start waiting for response
                            transcriptFound = true;  // Set flag to prevent further checks
                            displayTranscription(data.transcription);
                            
                            // Start the result generation phase
                            startResultGeneration();
                        } else if (data.status === 'complete' && data.result && data.result.transcription) {
                            // We have a result with transcription
                            transcriptFound = true;
                            displayTranscription(data.result.transcription);
                            
                            // Check if we also have a response
                            if (data.result.response) {
                                displayResponse(data.result);
                            } else {
                                startResultGeneration();
                            }
                        } else {
                            console.warn('[Voice] No transcription available after maximum checks');
                            showErrorMessage('Could not get voice transcription. Please try again.');
                            removeStatusIndicator();
                        }
                    })
                    .catch(error => {
                        console.error('[Voice] Error in fallback transcription check:', error);
                        showErrorMessage('Error retrieving voice transcription. Please try again.');
                        removeStatusIndicator();
                    });
                }
            })
            .catch(error => {
                console.error('[Voice] Error checking transcript file:', error);
                if (checkCount < maxChecks) {
                    // Schedule another check despite the error
                    setTimeout(checkTranscriptFile, checkInterval);
                } else {
                    showErrorMessage('Error retrieving voice transcription. Please try again.');
                    removeStatusIndicator();
                }
            });
        }
        
        // Start checking for the transcript file
        checkTranscriptFile();
    }
    
    // Display the transcription as a user message
    function displayTranscription(transcription) {
        console.log('[Voice] Displaying transcription:', transcription);
        
        // Remove the status indicator before showing transcription
        removeStatusIndicator();
        
        // Add the user's transcribed message
        if (transcription) {
            // Use the global addUserMessage function if available
            if (typeof window.addUserMessage === 'function') {
                console.log('[Voice] Using global addUserMessage function');
                window.addUserMessage(transcription);
            } else {
                console.log('[Voice] Using local addUserMessage implementation');
                const userMessageElement = document.createElement('div');
                userMessageElement.className = 'message user';
                userMessageElement.textContent = transcription;
                conversationContainer.appendChild(userMessageElement);
            }
            
            // Scroll to bottom
            conversationContainer.scrollTop = conversationContainer.scrollHeight;
        } else {
            console.warn('[Voice] Empty transcription received');
            showErrorMessage('Could not transcribe audio clearly. Please try again.');
        }
    }
    
    // ----- STEP 4: GENERATE RESULTS -----
    // Start the result generation phase
    function startResultGeneration() {
        console.log('[Voice] Starting result generation phase');
        showStatusIndicator('generating');
        
        // Start checking for results
        checkForResults();
    }
    
    // Check for results periodically
    function checkForResults() {
        console.log('[Voice] Checking for results');
        
        // Clear any existing check timer
        if (resultCheckTimer) {
            clearInterval(resultCheckTimer);
        }
        
        // Create a counter for check attempts
        let checkCount = 0;
        const maxChecks = 20; // Maximum number of checks (20 * 3s = 60 seconds max)
        let lastTranscript = null;
        
        // Check immediately, then every 3 seconds
        checkOnce();
        resultCheckTimer = setInterval(checkOnce, 3000);
        
        function checkOnce() {
            checkCount++;
            console.log(`[Voice] Result check ${checkCount}/${maxChecks}`);
            
            // Stop checking after maximum attempts
            if (checkCount >= maxChecks) {
                clearInterval(resultCheckTimer);
                resultCheckTimer = null;
                showErrorMessage('Response generation timed out. Please try asking again.');
                removeStatusIndicator();
                return;
            }
            
            fetch(`/api/check-processing/${sessionId}`)
            .then(response => response.json())
            .then(data => {
                console.log('[Voice] Result check response:', data);
                
                // Save the transcript we're seeing
                if (data.transcription) {
                    lastTranscript = data.transcription;
                }
                
                // Check if we have a full response with response field
                if (data.status === 'complete' && data.result && data.result.response) {
                    // We have the response, display it
                    clearInterval(resultCheckTimer);
                    resultCheckTimer = null;
                    
                    displayResponse(data.result);
                }
                // Check for CrewAI errors
                else if (data.result && data.result.error && data.result.error === "CrewAI processing error") {
                    console.log('[Voice] Detected CrewAI error, resetting system');
                    clearInterval(resultCheckTimer);
                    resultCheckTimer = null;
                    
                    // Try to reset the crew manager
                    fetch('/api/reset-crew', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                    .then(response => response.json())
                    .then(resetData => {
                        console.log('[Voice] Crew reset result:', resetData);
                    })
                    .catch(error => {
                        console.error('[Voice] Error resetting crew:', error);
                    });
                    
                    // Display the error response
                    displayResponse(data.result);
                }
                // If we've been checking for a while but only have a transcript, show it as final answer
                else if (checkCount >= 10 && lastTranscript) {
                    clearInterval(resultCheckTimer);
                    resultCheckTimer = null;
                    
                    // Create a simple result object with just the transcript
                    const fallbackResult = {
                        response: `I processed your query: "${lastTranscript}" but couldn't generate a complete response in time.`,
                        transcription: lastTranscript,
                        confidence: 0.5
                    };
                    
                    displayResponse(fallbackResult);
                }
            })
            .catch(error => {
                console.error('[Voice] Error checking for results:', error);
                
                // Stop checking after 3 consecutive errors
                if (checkCount > 3) {
                    clearInterval(resultCheckTimer);
                    resultCheckTimer = null;
                    showErrorMessage('Error retrieving response. Please try again.');
                    removeStatusIndicator();
                }
            });
        }
    }
    
    // Display the final response
    function displayResponse(result) {
        console.log('[Voice] Displaying response:', result);
        
        // Remove any status indicator
        removeStatusIndicator();
        
        // Display the bot's response
        if (result.response) {
            if (typeof window.addBotMessage === 'function') {
                console.log('[Voice] Using global addBotMessage function');
                
                // Format the response for dashboard display
                let botResponse = {
                    answer: typeof result.response === 'string' ? result.response : JSON.stringify(result.response),
                    method: "Based on voice query analysis.",
                    confidence: result.confidence || 0.8,
                    category: "Miscellaneous"
                };
                
                // If response is an object with answer field, use that structure
                if (typeof result.response === 'object' && result.response.answer) {
                    botResponse = result.response;
                }
                
                window.addBotMessage(botResponse);
            } else {
                console.log('[Voice] Using fallback display method');
                const botMessageElement = document.createElement('div');
                botMessageElement.className = 'message bot';
                botMessageElement.textContent = typeof result.response === 'string' ? 
                    result.response : JSON.stringify(result.response);
                conversationContainer.appendChild(botMessageElement);
            }
            
            // Scroll to bottom
            conversationContainer.scrollTop = conversationContainer.scrollHeight;
            
            // Clean up transcript file
            console.log('[Voice] Cleaning up transcript file');
            fetch('/api/cleanup-transcript', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => {
                console.log('[Voice] Cleanup response status:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('[Voice] Transcript cleanup result:', data);
            })
            .catch(error => {
                console.error('[Voice] Error cleaning up transcript:', error);
            });
            
            // Reset the global transcript variable (if any) by sending an extra request
            console.log('[Voice] Resetting global transcript');
            fetch('/api/reset-transcript', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .catch(error => {
                console.error('[Voice] Error resetting transcript:', error);
            });
        }
    }
    
    // ----- HELPER FUNCTIONS -----
    // Show status indicator in the conversation
    function showStatusIndicator(status) {
        console.log(`[Voice] Showing status indicator: ${status}`);
        
        // Remove any existing indicator first
        removeStatusIndicator();
        
        // Create indicator element
        const indicatorElement = document.createElement('div');
        indicatorElement.className = `message bot status-indicator ${status}`;
        indicatorElement.id = 'status-indicator';
        
        // Create pulsing dots
        const pulsingDots = document.createElement('div');
        pulsingDots.className = 'pulsing-dots';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'dot';
            pulsingDots.appendChild(dot);
        }
        
        // Set the status text
        const statusText = document.createElement('div');
        statusText.className = 'status-text';
        
        switch(status) {
            case 'recording':
                statusText.textContent = 'Recording your voice...';
                break;
            case 'transcribing':
                statusText.textContent = 'Transcribing audio...';
                break;
            case 'generating':
                statusText.textContent = 'Generating response...';
                break;
            default:
                statusText.textContent = 'Processing...';
                break;
        }
        
        // Add custom styling for different statuses
        if (!document.getElementById('status-styles')) {
            const style = document.createElement('style');
            style.id = 'status-styles';
            style.textContent = `
                .message.bot.status-indicator {
                    display: flex;
                    align-items: center;
                    padding: 10px 16px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                }
                .message.bot.status-indicator.recording {
                    background-color: #ffeded;
                    border-left: 3px solid #e74c3c;
                }
                .message.bot.status-indicator.transcribing {
                    background-color: #f0f7ff;
                    border-left: 3px solid #3498db;
                }
                .message.bot.status-indicator.generating {
                    background-color: #f7f9ff;
                    border-left: 3px solid #9b59b6;
                }
                .pulsing-dots {
                    display: flex;
                    margin-left: 10px;
                }
                .dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background-color: #555;
                    margin-right: 4px;
                    animation: pulse 1.5s infinite;
                }
                .dot:nth-child(2) {
                    animation-delay: 0.2s;
                }
                .dot:nth-child(3) {
                    animation-delay: 0.4s;
                }
                @keyframes pulse {
                    0% { transform: scale(0.8); opacity: 0.5; }
                    50% { transform: scale(1.2); opacity: 1; }
                    100% { transform: scale(0.8); opacity: 0.5; }
                }
            `;
            document.head.appendChild(style);
        }
        
        // Assemble the indicator
        indicatorElement.appendChild(statusText);
        indicatorElement.appendChild(pulsingDots);
        
        // Add to conversation
        if (conversationContainer) {
            conversationContainer.appendChild(indicatorElement);
            conversationContainer.scrollTop = conversationContainer.scrollHeight;
        }
    }
    
    // Remove status indicator
    function removeStatusIndicator() {
        console.log('[Voice] Removing status indicator');
        const indicator = document.getElementById('status-indicator');
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    }
    
    // Show error message in the conversation
    function showErrorMessage(message) {
        console.log('[Voice] Showing error message:', message);
        
        const errorElement = document.createElement('div');
        errorElement.className = 'message bot error';
        errorElement.textContent = message;
        
        if (conversationContainer) {
            conversationContainer.appendChild(errorElement);
            conversationContainer.scrollTop = conversationContainer.scrollHeight;
        }
    }
}

// Make sure transcription results can be tracked similar to text queries
function trackQuery(query, category, value, event) {
    console.log('[Voice] Tracking query:', query, 'Category:', category, 'Value:', value);
    // Check if window.trackQuery is available (from tracking.js)
    if (typeof window.trackQuery === 'function') {
        console.log('[Voice] Using global trackQuery function');
        window.trackQuery(query, category, value, event);
    } else {
        console.error('[Voice] trackQuery function not available');
    }
}

// Export to make functions available to other modules
window.VoiceHandler = {
    trackQuery: trackQuery
};

console.log('[Voice] Voice handler module loaded');
