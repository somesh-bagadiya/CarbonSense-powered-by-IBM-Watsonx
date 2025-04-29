// CarbonSense Voice Handler JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('[Voice] Initializing voice recording module');
    
    // Check if AGENT_MESSAGES is loaded
    if (window.AGENT_MESSAGES) {
        console.log('[Voice] AGENT_MESSAGES is available:', Object.keys(window.AGENT_MESSAGES).length, 'agents defined');
    } else {
        console.warn('[Voice] AGENT_MESSAGES not found, adding script to load it');
        
        // Add the agent-messages.js script if not already added
        if (!document.querySelector('script[src*="agent-messages.js"]')) {
            const script = document.createElement('script');
            script.src = '/static/js/agent-messages.js';
            script.onload = function() {
                console.log('[Voice] AGENT_MESSAGES script loaded successfully');
            };
            script.onerror = function() {
                console.error('[Voice] Failed to load AGENT_MESSAGES script');
            };
            document.head.appendChild(script);
        }
    }
    
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
        
        // Create thinking container similar to dashboard.js
        const thinkingContainer = document.createElement('div');
        thinkingContainer.className = 'thinking-container';
        
        // Create header for thinking container
        const thinkingHeader = document.createElement('div');
        thinkingHeader.className = 'thinking-header';
        thinkingHeader.innerHTML = '<i class="fas fa-microphone"></i> Recording...';
        thinkingContainer.appendChild(thinkingHeader);
        
        // Create thoughts area with no max-height limit
        const thoughtsArea = document.createElement('div');
        thoughtsArea.className = 'thoughts-area';
        thoughtsArea.style.maxHeight = 'none'; // Remove max-height limit
        thinkingContainer.appendChild(thoughtsArea);
        
        // Create a single reusable thought element
        const thoughtElement = document.createElement('div');
        thoughtElement.className = 'thought';
        thoughtElement.innerHTML = `
            <span class="thought-icon"><i class="fas fa-microphone-alt"></i></span>
            <span class="thought-content">Listening to your voice query...</span>
        `;
        thoughtsArea.appendChild(thoughtElement);
        
        // Add to conversation
        conversationContainer.appendChild(thinkingContainer);
        ensureScrolling();
        
        // Start timer to update recording time display and the thought element
        recordingTimer = setInterval(function() {
            recordingSeconds++;
            recordingTimeElement.textContent = recordingSeconds;
            
            // Update the recording thought
            const contentElement = thoughtElement.querySelector('.thought-content');
            if (contentElement) {
                contentElement.textContent = `Listening to your voice query... (${recordingSeconds}s)`;
            }
            
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
                
                // Update the thought with session info
                const contentElement = thoughtElement.querySelector('.thought-content');
                if (contentElement) {
                    contentElement.textContent = `Listening to your voice query... (Session: ${sessionId})`;
                }
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
            console.log('[Voice] Error occurred, removing any thinking containers');
            // Remove any thinking containers
            const thinkingContainers = document.querySelectorAll('.thinking-container');
            thinkingContainers.forEach(container => container.remove());
            return;
        }
        
        // Find the thinking container and update it for transcription phase
        const thinkingContainer = document.querySelector('.thinking-container');
        if (thinkingContainer) {
            // Update the header text
            const thinkingHeader = thinkingContainer.querySelector('.thinking-header');
            if (thinkingHeader) {
                thinkingHeader.innerHTML = '<i class="fas fa-language"></i> Transcribing...';
            }
            
            // Update or add a thought for transcription
            const thoughtsArea = thinkingContainer.querySelector('.thoughts-area');
            if (thoughtsArea) {
                // Make sure thoughts area has no max-height limit
                thoughtsArea.style.maxHeight = 'none';
                
                // Check if there's already a thought element we can reuse
                let thoughtElement = thoughtsArea.querySelector('.thought');
                
                // If no existing thought, create a new one
                if (!thoughtElement) {
                    thoughtElement = document.createElement('div');
                    thoughtElement.className = 'thought';
                    thoughtElement.innerHTML = `
                        <span class="thought-icon"><i class="fas fa-headphones"></i></span>
                        <span class="thought-content">Processing audio to text...</span>
                    `;
                    thoughtsArea.appendChild(thoughtElement);
                } else {
                    // Update existing thought
                    const iconElement = thoughtElement.querySelector('.thought-icon');
                    if (iconElement) {
                        iconElement.innerHTML = '<i class="fas fa-headphones"></i>';
                    }
                    
                    const contentElement = thoughtElement.querySelector('.thought-content');
                    if (contentElement) {
                        contentElement.textContent = 'Processing audio to text...';
                    }
                }
                
                // Add a new action thought
                const actionThought = document.createElement('div');
                actionThought.className = 'thought action';
                actionThought.innerHTML = `
                    <span class="thought-icon"><i class="fas fa-cog fa-spin"></i></span>
                    <span class="thought-content">Converting speech to text with IBM Watson...</span>
                `;
                thoughtsArea.appendChild(actionThought);
                
                // Ensure scrolling
                ensureScrolling();
            }
        } else {
            // Fallback to old method if thinking container not found
            showStatusIndicator('transcribing');
        }
            
        // If we have a session, proceed to transcription
        if (sessionId) {
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
                
                // Remove thinking container and show error
                if (thinkingContainer) {
                    thinkingContainer.remove();
                }
                
                showErrorMessage('Error processing your voice. Please try again.');
            });
        }
    }
    
    // ----- STEP 3: FETCH TRANSCRIPTION AFTER DELAY -----
    // Fetch transcription after delay
    function fetchTranscription() {
        console.log('[Voice] Fetching transcription after delay');
        
        // Implement polling for the transcript file
        let checkCount = 0;
        const maxChecks = 40; // Maximum number of checks
        const checkInterval = 1500; // Check every 1.5 seconds
        let transcriptFound = false;  // Flag to track if transcript was found
        
        // Find the thinking container
        const thinkingContainer = document.querySelector('.thinking-container');
        const thoughtsArea = thinkingContainer ? thinkingContainer.querySelector('.thoughts-area') : null;
        
        // Update the thinking container if it exists
        if (thinkingContainer && thinkingContainer.querySelector('.thinking-header')) {
            thinkingContainer.querySelector('.thinking-header').innerHTML = 
                '<i class="fas fa-file-alt"></i> Fetching Transcription...';
        }
        
        // Add a new thought if we have a thoughts area
        if (thoughtsArea) {
            const newThought = document.createElement('div');
            newThought.className = 'thought action';
            newThought.innerHTML = `
                <span class="thought-icon"><i class="fas fa-search"></i></span>
                <span class="thought-content">Checking for completed transcription...</span>
            `;
            thoughtsArea.appendChild(newThought);
            thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
        }
        
        function checkTranscriptFile() {
            // If transcript already found, don't check again
            if (transcriptFound) {
                console.log('[Voice] Transcript already found, skipping check');
                return;
            }
            
            checkCount++;
            console.log(`[Voice] Checking for transcript (attempt ${checkCount}/${maxChecks})`);
            
            // Update thought if it exists
            if (thoughtsArea) {
                const lastThought = thoughtsArea.lastElementChild;
                if (lastThought && lastThought.querySelector('.thought-content')) {
                    lastThought.querySelector('.thought-content').textContent = 
                        `Checking for completed transcription (attempt ${checkCount}/${maxChecks})...`;
                }
            }
            
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
                    
                    // Add success thought
                    if (thoughtsArea) {
                        const successThought = document.createElement('div');
                        successThought.className = 'thought complete';
                        successThought.innerHTML = `
                            <span class="thought-icon"><i class="fas fa-check-circle"></i></span>
                            <span class="thought-content">Transcription completed: "${data.transcript}"</span>
                        `;
                        thoughtsArea.appendChild(successThought);
                        thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                    }
                    
                    displayTranscription(data.transcript);
                    
                    // Start the result generation phase with a new thinking container
                    startResultGeneration(data.transcript);
                    
                    // Clear any timers and return to stop further checks
                    if (resultCheckTimer) {
                        clearInterval(resultCheckTimer);
                        resultCheckTimer = null;
                    }
                    return;
                } else if (checkCount < maxChecks) {
                    // Schedule another check
                    setTimeout(checkTranscriptFile, checkInterval);
                } else {
                    // Fallback to the original check-processing endpoint
                    console.log('[Voice] Falling back to check-processing endpoint');
                    
                    // Add fallback thought
                    if (thoughtsArea) {
                        const fallbackThought = document.createElement('div');
                        fallbackThought.className = 'thought action';
                        fallbackThought.innerHTML = `
                            <span class="thought-icon"><i class="fas fa-redo"></i></span>
                            <span class="thought-content">Falling back to processing status check...</span>
                        `;
                        thoughtsArea.appendChild(fallbackThought);
                        thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                    }
                    
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
                            
                            // Add success thought
                            if (thoughtsArea) {
                                const successThought = document.createElement('div');
                                successThought.className = 'thought complete';
                                successThought.innerHTML = `
                                    <span class="thought-icon"><i class="fas fa-check-circle"></i></span>
                                    <span class="thought-content">Transcription found: "${data.transcription}"</span>
                                `;
                                thoughtsArea.appendChild(successThought);
                                thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                            }
                            
                            displayTranscription(data.transcription);
                            
                            // Start the result generation phase with a new thinking container
                            startResultGeneration(data.transcription);
                            
                            // Clear any timers and return to stop further checks
                            if (resultCheckTimer) {
                                clearInterval(resultCheckTimer);
                                resultCheckTimer = null;
                            }
                            return;
                        } else if (data.status === 'complete' && data.result && data.result.transcription) {
                            // We have a result with transcription
                            transcriptFound = true;
                            
                            // Add success thought
                            if (thoughtsArea) {
                                const successThought = document.createElement('div');
                                successThought.className = 'thought complete';
                                successThought.innerHTML = `
                                    <span class="thought-icon"><i class="fas fa-check-circle"></i></span>
                                    <span class="thought-content">Transcription found: "${data.result.transcription}"</span>
                                `;
                                thoughtsArea.appendChild(successThought);
                                thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                            }
                            
                            displayTranscription(data.result.transcription);
                            
                            // Check if we also have a response
                            if (data.result.response) {
                                displayResponse(data.result);
                            } else {
                                // Start the result generation phase with a new thinking container
                                startResultGeneration(data.result.transcription);
                            }
                            
                            // Clear any timers and return to stop further checks
                            if (resultCheckTimer) {
                                clearInterval(resultCheckTimer);
                                resultCheckTimer = null;
                            }
                            return;
                        } else {
                            console.warn('[Voice] No transcription available after maximum checks');
                            
                            // Add error thought
                            if (thoughtsArea) {
                                const errorThought = document.createElement('div');
                                errorThought.className = 'thought error';
                                errorThought.innerHTML = `
                                    <span class="thought-icon"><i class="fas fa-exclamation-triangle"></i></span>
                                    <span class="thought-content">No transcription available after maximum checks</span>
                                `;
                                thoughtsArea.appendChild(errorThought);
                                thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                                
                                // Remove thinking container after delay
                                setTimeout(() => {
                                    if (thinkingContainer) {
                                        thinkingContainer.remove();
                                    }
                                }, 1000);
                            }
                            
                            showErrorMessage('Could not get voice transcription. Please try again.');
                        }
                    })
                    .catch(error => {
                        console.error('[Voice] Error in fallback transcription check:', error);
                        
                        // Add error thought
                        if (thoughtsArea) {
                            const errorThought = document.createElement('div');
                            errorThought.className = 'thought error';
                            errorThought.innerHTML = `
                                <span class="thought-icon"><i class="fas fa-exclamation-triangle"></i></span>
                                <span class="thought-content">Error in fallback check: ${error.message}</span>
                            `;
                            thoughtsArea.appendChild(errorThought);
                            thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                            
                            // Remove thinking container after delay
                            setTimeout(() => {
                                if (thinkingContainer) {
                                    thinkingContainer.remove();
                                }
                            }, 1000);
                        }
                        
                        showErrorMessage('Error retrieving voice transcription. Please try again.');
                    });
                }
            })
            .catch(error => {
                console.error('[Voice] Error checking transcript file:', error);
                if (checkCount < maxChecks) {
                    // Schedule another check despite the error
                    setTimeout(checkTranscriptFile, checkInterval);
                } else {
                    // Add error thought
                    if (thoughtsArea) {
                        const errorThought = document.createElement('div');
                        errorThought.className = 'thought error';
                        errorThought.innerHTML = `
                            <span class="thought-icon"><i class="fas fa-exclamation-triangle"></i></span>
                            <span class="thought-content">Error checking transcript: ${error.message}</span>
                        `;
                        thoughtsArea.appendChild(errorThought);
                        thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                        
                        // Remove thinking container after delay
                        setTimeout(() => {
                            if (thinkingContainer) {
                                thinkingContainer.remove();
                            }
                        }, 1000);
                    }
                    
                    showErrorMessage('Error retrieving voice transcription. Please try again.');
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
            
            // Ensure scrolling
            ensureScrolling();
        } else {
            console.warn('[Voice] Empty transcription received');
            showErrorMessage('Could not transcribe audio clearly. Please try again.');
        }
    }
    
    // ----- STEP 4: GENERATE RESULTS -----
    // Start the result generation phase with a new thinking container
    function startResultGeneration(transcript) {
        console.log('[Voice] Starting result generation phase with new thinking container');
        
        // Create a new thinking container for response generation
        const newThinkingContainer = document.createElement('div');
        newThinkingContainer.className = 'thinking-container';
        
        // Create header for the new thinking container
        const newHeader = document.createElement('div');
        newHeader.className = 'thinking-header';
        newHeader.innerHTML = '<i class="fas fa-robot"></i> Generating Response...';
        newThinkingContainer.appendChild(newHeader);
        
        // Create thoughts area
        const newThoughtsArea = document.createElement('div');
        newThoughtsArea.className = 'thoughts-area';
        newThoughtsArea.style.maxHeight = 'none'; // Remove max height to allow full display
        newThinkingContainer.appendChild(newThoughtsArea);
            
            // Add a new thought about generation
                const generationThought = document.createElement('div');
                generationThought.className = 'thought action';
                generationThought.innerHTML = `
                    <span class="thought-icon"><i class="fas fa-cog fa-spin"></i></span>
                    <span class="thought-content">Analyzing transcript and generating a thoughtful response...</span>
                `;
        newThoughtsArea.appendChild(generationThought);
        
        // Add initial thought about the transcript
        if (transcript) {
            const transcriptThought = document.createElement('div');
            transcriptThought.className = 'thought';
            transcriptThought.innerHTML = `
                <span class="thought-icon"><i class="fas fa-comment"></i></span>
                <span class="thought-content">Processing query: "${transcript}"</span>
            `;
            newThoughtsArea.appendChild(transcriptThought);
        }
        
        // Add to conversation
        conversationContainer.appendChild(newThinkingContainer);
        
        // Ensure scrolling
        ensureScrolling();
        
        // Start checking for results
        checkForResults(newThinkingContainer);
    }
    
    // Check for results periodically
    function checkForResults(thinkingContainer) {
        console.log('[Voice] Starting result check process');
        
        // Clear any existing check timer
        if (resultCheckTimer) {
            clearInterval(resultCheckTimer);
            resultCheckTimer = null;
        }
        
        // Create request ID for tracking
        const requestId = `query_${Date.now()}`;
        
        // Create counter variables for checks
        let checkCount = 0;
        const maxChecks = 60; // 3 minutes total with 3-second intervals
        let lastTranscript = null;
        let lastSeenThoughtIndex = -1;
        let isPolling = false;
        let hasShownWaitingMessage = false;
        
        // After 30 seconds (10 checks), show a waiting message
        const showWaitingMessageAfter = 10;
        
        // Use provided thinking container or find existing one
        let thoughtsArea;
        let thinkingHeader;
        
        if (!thinkingContainer) {
            // Find existing thinking container
            thinkingContainer = document.querySelector('.thinking-container:last-child');
        }
        
        if (thinkingContainer) {
            // Get existing elements
            thinkingHeader = thinkingContainer.querySelector('.thinking-header');
            thoughtsArea = thinkingContainer.querySelector('.thoughts-area');
            
            // Update header if needed
            if (thinkingHeader) {
                thinkingHeader.innerHTML = '<i class="fas fa-robot"></i> Processing with CrewAI...';
            }
            
            // Add a processing thought if needed
            if (thoughtsArea && thoughtsArea.children.length === 0) {
                const initialThought = document.createElement('div');
                initialThought.className = 'thought start';
                initialThought.innerHTML = `
                    <span class="thought-icon"><i class="fas fa-play"></i></span>
                    <span class="thought-content">Starting carbon footprint analysis...</span>
                `;
                thoughtsArea.appendChild(initialThought);
            }
        } else {
            // Create new thinking container if none was provided or found
            thinkingContainer = document.createElement('div');
            thinkingContainer.className = 'thinking-container';
            
            // Create header for thinking container
            thinkingHeader = document.createElement('div');
            thinkingHeader.className = 'thinking-header';
            thinkingHeader.innerHTML = '<i class="fas fa-robot"></i> Processing with CrewAI...';
            thinkingContainer.appendChild(thinkingHeader);
            
            // Create thoughts area
            thoughtsArea = document.createElement('div');
            thoughtsArea.className = 'thoughts-area';
            thinkingContainer.appendChild(thoughtsArea);
            
            // Create initial thought
            const initialThought = document.createElement('div');
            initialThought.className = 'thought start';
            initialThought.innerHTML = `
                <span class="thought-icon"><i class="fas fa-play"></i></span>
                <span class="thought-content">Starting carbon footprint analysis...</span>
            `;
            thoughtsArea.appendChild(initialThought);
            
            // Add to conversation
            conversationContainer.appendChild(thinkingContainer);
        }
        
        // Ensure container is visible and scrolled into view
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
        
        // Remove any status indicators since we're using the thinking container
        removeStatusIndicator();
        
        // Start thought polling
        const thoughtPollingId = startThoughtPolling();
        
        // Function to start thought polling
        function startThoughtPolling() {
            console.log('[Voice] Starting thought polling process');
            pollForThoughts();
            return setInterval(pollForThoughts, 3000);
        }
        
        // Function to poll for thoughts
        async function pollForThoughts() {
            // Prevent multiple concurrent polling
            if (isPolling) return;
            
            isPolling = true;
            checkCount++;
            
            try {
                console.log(`[Voice] Polling for thoughts (check #${checkCount}/${maxChecks})`);
                
                // Make direct fetch request to backend
                const response = await fetch('/api/check-thoughts');
                if (!response.ok) {
                    throw new Error(`HTTP error: ${response.status}`);
                }
                
                const thoughtsData = await response.json();
                console.log('[Voice] Received thoughts data:', thoughtsData);
                
                // Process the thoughts if we have any
                if (thoughtsData && thoughtsData.thoughts && thoughtsData.thoughts.length > 0) {
                    // Process only new thoughts
                    const newThoughts = thoughtsData.thoughts.slice(lastSeenThoughtIndex + 1);
                    
                    if (newThoughts.length > 0) {
                        console.log('[Voice] Found new thoughts:', newThoughts.length);
                        
                        // If there are existing thoughts, replace the content of the last one
                        // instead of adding many new thoughts
                        if (thoughtsArea.children.length > 0 && newThoughts.length > 0) {
                            // Keep only the first thought for context
                            while (thoughtsArea.children.length > 1) {
                                thoughtsArea.removeChild(thoughtsArea.lastChild);
                            }
                            
                            // Get the most recent and relevant thought
                            const latestThought = newThoughts[newThoughts.length - 1];
                            
                            // Create or update thought element
                            let thoughtElement;
                            if (thoughtsArea.children.length > 0) {
                                // Update existing thought
                                thoughtElement = thoughtsArea.lastChild;
                            } else {
                                // Create new thought if needed
                                thoughtElement = document.createElement('div');
                                thoughtElement.className = `thought ${latestThought.type || 'thought'}`;
                                thoughtsArea.appendChild(thoughtElement);
                            }
                            
                            // Add icon based on type
                            let icon = '';
                            switch(latestThought.type) {
                                case 'thought':
                                case 'thinking':
                                    icon = '<i class="fas fa-brain"></i>';
                                    break;
                                case 'agent_step':
                                    icon = '<i class="fas fa-cog fa-spin"></i>';
                                    break;
                                case 'agent_detail':
                                    icon = '<i class="fas fa-search"></i>';
                                    break;
                                case 'error':
                                    icon = '<i class="fas fa-exclamation-triangle"></i>';
                                    break;
                                case 'complete':
                                case 'completion':
                                    icon = '<i class="fas fa-check-circle"></i>';
                                    break;
                                case 'start':
                                    icon = '<i class="fas fa-play"></i>';
                                    break;
                                default:
                                    icon = '<i class="fas fa-comment"></i>';
                            }
                            
                            // For agent steps, only show the agent name
                            if (latestThought.type === 'agent_step' && latestThought.content) {
                                // Extract just the agent name from the content
                                let displayContent = latestThought.content;
                                
                                // Look for patterns like "Processing with X..." or "Agent: X"
                                const agentNameMatch = displayContent.match(/Processing with ([^\.]+)\.{3}/) ||
                                                     displayContent.match(/Agent:\s*([^\n]+)/);
                                
                                if (agentNameMatch && agentNameMatch[1]) {
                                    // Just use the matched agent name with "Processing with..." prefix
                                    displayContent = `Processing with ${agentNameMatch[1]}...`;
                                } else {
                                    // If we have an agent_ops_agent_name field, use that
                                    const agentNameSearch = displayContent.match(/agent_ops_agent_name='([^']+)'/);
                                    if (agentNameSearch && agentNameSearch[1]) {
                                        displayContent = `Processing with ${agentNameSearch[1].trim()}...`;
                                    } else {
                                        // Fall back to just the first line of content
                                        const firstLine = displayContent.split('\n')[0];
                                        if (firstLine && firstLine.length > 0) {
                                            displayContent = firstLine;
                                        }
                                        
                                        // If it's still too long, truncate it
                                        if (displayContent.length > 60) {
                                            displayContent = displayContent.substring(0, 60) + '...';
                                        }
                                    }
                                }
                                
                                // Simple text content
                                thoughtElement.innerHTML = `
                                    <span class="thought-icon">${icon}</span>
                                    <span class="thought-content">${displayContent}</span>
                                `;
                            } else {
                                // Check if this thought contains code or should preserve formatting
                                const shouldPreserveFormatting = 
                                    latestThought.preserve_formatting || 
                                    (latestThought.content && latestThought.content.includes('```')) || 
                                    latestThought.type === 'agent_detail';
                                
                                if (shouldPreserveFormatting) {
                                    // Format with HTML for code blocks
                                    thoughtElement.innerHTML = `
                                        <span class="thought-icon">${icon}</span>
                                        <div class="thought-content formatted"></div>
                                    `;
                                    
                                    // Format the content
                                    let formattedContent = latestThought.content || 'No content available';
                                    
                                    // Replace code blocks
                                    if (formattedContent.includes('```')) {
                                        // Replace markdown code blocks with HTML
                                        formattedContent = formattedContent.replace(/```(\w*)\n([\s\S]*?)```/g, 
                                            '<pre class="code-block"><code>$2</code></pre>');
                                    }
                                    
                                    // Replace newlines with <br> tags for proper display
                                    formattedContent = formattedContent.replace(/\n/g, '<br>');
                                    
                                    thoughtElement.querySelector('.thought-content').innerHTML = formattedContent;
                                } else {
                                    // Simple text content
                                    thoughtElement.innerHTML = `
                                        <span class="thought-icon">${icon}</span>
                                        <span class="thought-content">${latestThought.content || 'No content available'}</span>
                                    `;
                                }
                            }
                            
                            // Update class to match the thought type
                            thoughtElement.className = `thought ${latestThought.type || 'thought'}`;
                        } else {
                            // If no existing thoughts, process normally
                        newThoughts.forEach(thought => {
                            // Create a new thought element
                            const thoughtElement = document.createElement('div');
                            thoughtElement.className = `thought ${thought.type || 'thought'}`;
                            
                            // Add icon based on type
                            let icon = '';
                            switch(thought.type) {
                                case 'thought':
                                case 'thinking':
                                    icon = '<i class="fas fa-brain"></i>';
                                    break;
                                case 'agent_step':
                                    icon = '<i class="fas fa-cog fa-spin"></i>';
                                    break;
                                case 'agent_detail':
                                    icon = '<i class="fas fa-search"></i>';
                                    break;
                                case 'error':
                                    icon = '<i class="fas fa-exclamation-triangle"></i>';
                                    break;
                                case 'complete':
                                case 'completion':
                                    icon = '<i class="fas fa-check-circle"></i>';
                                    break;
                                case 'start':
                                    icon = '<i class="fas fa-play"></i>';
                                    break;
                                default:
                                    icon = '<i class="fas fa-comment"></i>';
                            }
                            
                                // For agent steps, only show the agent name
                                if (thought.type === 'agent_step' && thought.content) {
                                    // Extract just the agent name from the content
                                    let displayContent = thought.content;
                                    
                                    // Look for patterns like "Processing with X..." or "Agent: X"
                                    const agentNameMatch = displayContent.match(/Processing with ([^\.]+)\.{3}/) ||
                                                         displayContent.match(/Agent:\s*([^\n]+)/);
                                    
                                    if (agentNameMatch && agentNameMatch[1]) {
                                        // Just use the matched agent name with "Processing with..." prefix
                                        displayContent = `Processing with ${agentNameMatch[1]}...`;
                                    } else {
                                        // If we have an agent_ops_agent_name field, use that
                                        const agentNameSearch = displayContent.match(/agent_ops_agent_name='([^']+)'/);
                                        if (agentNameSearch && agentNameSearch[1]) {
                                            displayContent = `Processing with ${agentNameSearch[1].trim()}...`;
                                        } else {
                                            // Fall back to just the first line of content
                                            const firstLine = displayContent.split('\n')[0];
                                            if (firstLine && firstLine.length > 0) {
                                                displayContent = firstLine;
                                            }
                                            
                                            // If it's still too long, truncate it
                                            if (displayContent.length > 60) {
                                                displayContent = displayContent.substring(0, 60) + '...';
                                            }
                                        }
                                    }
                                    
                                    // Simple text content
                                    thoughtElement.innerHTML = `
                                        <span class="thought-icon">${icon}</span>
                                        <span class="thought-content">${displayContent}</span>
                                    `;
                                } else {
                            // Check if this thought contains code or should preserve formatting
                            const shouldPreserveFormatting = 
                                thought.preserve_formatting || 
                                (thought.content && thought.content.includes('```')) || 
                                thought.type === 'agent_detail';
                            
                            if (shouldPreserveFormatting) {
                                // Format with HTML for code blocks
                                thoughtElement.innerHTML = `
                                    <span class="thought-icon">${icon}</span>
                                    <div class="thought-content formatted"></div>
                                `;
                                
                                // Format the content
                                let formattedContent = thought.content || 'No content available';
                                
                                // Replace code blocks
                                if (formattedContent.includes('```')) {
                                    // Replace markdown code blocks with HTML
                                    formattedContent = formattedContent.replace(/```(\w*)\n([\s\S]*?)```/g, 
                                        '<pre class="code-block"><code>$2</code></pre>');
                                }
                                
                                // Replace newlines with <br> tags for proper display
                                formattedContent = formattedContent.replace(/\n/g, '<br>');
                                
                                thoughtElement.querySelector('.thought-content').innerHTML = formattedContent;
                            } else {
                                // Simple text content
                                thoughtElement.innerHTML = `
                                    <span class="thought-icon">${icon}</span>
                                    <span class="thought-content">${thought.content || 'No content available'}</span>
                                `;
                                    }
                            }
                            
                            // Add the thought to the container
                            thoughtsArea.appendChild(thoughtElement);
                        });
                        }
                        
                        // Update last seen thought index
                        lastSeenThoughtIndex = thoughtsData.thoughts.length - 1;
                        
                        // Ensure scrolling after updating thoughts
                        ensureScrolling();
                    }
                    
                    // Check for completion status
                    if (thoughtsData.status === 'complete' || thoughtsData.status === 'COMPLETE') {
                        console.log('[Voice] Thought process complete');
                        
                        // Add final thought if needed
                        if (!thoughtsArea.querySelector('.thought.complete')) {
                            const completeThought = document.createElement('div');
                            completeThought.className = 'thought complete';
                            completeThought.innerHTML = `
                                <span class="thought-icon"><i class="fas fa-check-circle"></i></span>
                                <span class="thought-content">Analysis complete! Preparing final response...</span>
                            `;
                            thoughtsArea.appendChild(completeThought);
                            thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                        }
                        
                        // Update header
                        if (thinkingHeader) {
                            thinkingHeader.innerHTML = '<i class="fas fa-check-circle"></i> Analysis Complete!';
                        }
                        
                        // Look for completion thoughts that contain final answer
                        const completionThoughts = thoughtsData.thoughts.filter(t => 
                            t.type === 'completion' || 
                            (t.type === 'complete' && t.content && t.content.includes('answer'))
                        );
                        
                        if (completionThoughts.length > 0) {
                            // Find the most recent completion thought
                            const latestCompletion = completionThoughts.reduce((latest, current) => {
                                if (!latest.timestamp) return current;
                                return current.timestamp > latest.timestamp ? current : latest;
                            }, {});
                            
                            // Try to extract structured data from completion thought
                            let result = null;
                            
                            try {
                                if (latestCompletion.content) {
                                    // Try to parse JSON data from content
                                    const jsonMatch = latestCompletion.content.match(/```json\s*([\s\S]*?)\s*```/);
                                    if (jsonMatch && jsonMatch[1]) {
                                        const parsedJson = JSON.parse(jsonMatch[1]);
                                        if (isValidCarbonResponse(parsedJson)) {
                                            result = {
                                                response: parsedJson,
                                                transcription: lastTranscript || ''
                                            };
                                        }
                                    } else {
                                        // If no JSON block, just use the content as plain text
                                        result = {
                                            response: {
                                                answer: latestCompletion.content.replace(/^Answer:\s*/i, ''),
                                                method: "CrewAI Analysis",
                                                confidence: 0.8,
                                                category: "Analysis"
                                            },
                                            transcription: lastTranscript || ''
                                        };
                                    }
                                }
                            } catch (e) {
                                console.error('[Voice] Error parsing completion thought:', e);
                            }
                            
                            // Stop polling and display result if we have one
                            if (result) {
                                clearInterval(thoughtPollingId);
                                
                                // Collapse thinking container
                                thinkingContainer.classList.add('collapsed');
                                
                                // Create a toggle button
                                const toggleButton = document.createElement('button');
                                toggleButton.className = 'toggle-thinking-btn';
                                toggleButton.innerHTML = '<i class="fas fa-lightbulb"></i> Show Thinking';
                                toggleButton.addEventListener('click', () => {
                                    thinkingContainer.classList.toggle('collapsed');
                                    toggleButton.innerHTML = thinkingContainer.classList.contains('collapsed') ?
                                        '<i class="fas fa-lightbulb"></i> Show Thinking' :
                                        '<i class="fas fa-lightbulb"></i> Hide Thinking';
                                });
                                
                                // Add toggle button to thinking header
                                if (thinkingHeader) {
                                    thinkingHeader.appendChild(toggleButton);
                                }
                                
                                // Display the response
                                displayResponse(result);
                                return;
                            }
                        }
                        
                        // If we didn't find a usable result but status is complete,
                        // try the regular endpoint once more
                        checkOnce(true);
                    } 
                    // Check for error status
                    else if (thoughtsData.status === 'error' || thoughtsData.status === 'ERROR') {
                        console.log('[Voice] Thought process error');
                        
                        // Look for error thoughts
                        const errorThoughts = thoughtsData.thoughts.filter(t => t.type === 'error');
                        let errorMessage = 'An error occurred during processing.';
                        
                        if (errorThoughts.length > 0) {
                            // Get the most recent error thought
                            const latestError = errorThoughts.reduce((latest, current) => {
                                if (!latest.timestamp) return current;
                                return current.timestamp > latest.timestamp ? current : latest;
                            }, {});
                            
                            errorMessage = latestError.content || errorMessage;
                        }
                        
                        // Add error thought if needed
                        if (!thoughtsArea.querySelector('.thought.error')) {
                            const errorThought = document.createElement('div');
                            errorThought.className = 'thought error';
                            errorThought.innerHTML = `
                                <span class="thought-icon"><i class="fas fa-exclamation-triangle"></i></span>
                                <span class="thought-content">${errorMessage}</span>
                            `;
                            thoughtsArea.appendChild(errorThought);
                            thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                        }
                        
                        // Update header
                        if (thinkingHeader) {
                            thinkingHeader.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error Occurred';
                        }
                        
                        // Stop polling
                        clearInterval(thoughtPollingId);
                        
                        // Display error response
                        const errorResult = {
                            response: {
                                answer: errorMessage,
                                method: "Error during processing",
                                confidence: 0.1,
                                category: "Error"
                            },
                            transcription: lastTranscript || ""
                        };
                        
                        // Collapse thinking container
                        thinkingContainer.classList.add('collapsed');
                        
                        // Create a toggle button
                        const toggleButton = document.createElement('button');
                        toggleButton.className = 'toggle-thinking-btn';
                        toggleButton.innerHTML = '<i class="fas fa-lightbulb"></i> Show Thinking';
                        toggleButton.addEventListener('click', () => {
                            thinkingContainer.classList.toggle('collapsed');
                            toggleButton.innerHTML = thinkingContainer.classList.contains('collapsed') ?
                                '<i class="fas fa-lightbulb"></i> Show Thinking' :
                                '<i class="fas fa-lightbulb"></i> Hide Thinking';
                        });
                        
                        // Add toggle button to thinking header
                        if (thinkingHeader) {
                            thinkingHeader.appendChild(toggleButton);
                        }
                        
                        displayResponse(errorResult);
                        return;
                    }
                } else {
                    console.log('[Voice] No thoughts available yet');
                    
                    // If we've been waiting a while, show a waiting message
                    if (!hasShownWaitingMessage && checkCount >= showWaitingMessageAfter) {
                        hasShownWaitingMessage = true;
                        const waitingThought = document.createElement('div');
                        waitingThought.className = 'thought action';
                        waitingThought.innerHTML = `
                            <span class="thought-icon"><i class="fas fa-clock"></i></span>
                            <span class="thought-content">This is taking longer than usual. Still processing your request...</span>
                        `;
                        thoughtsArea.appendChild(waitingThought);
                        thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                    }
                    
                    // Check the regular endpoint after a few attempts with no thoughts
                    // This helps us get results faster if the thoughts API isn't returning data
                    if (checkCount % 5 === 0) {  // Check every 15 seconds (5 polling intervals)
                        console.log('[Voice] Checking regular endpoint in parallel for faster response');
                        checkOnce(false);
                    }
                    
                    // After max checks with no thoughts, try the regular endpoint once more as fallback
                    if (checkCount >= maxChecks) {
                        checkOnce(true);
                        clearInterval(thoughtPollingId);
                    }
                }
            } catch (error) {
                console.error('[Voice] Error fetching thoughts:', error);
                
                // After several consecutive errors, try the regular endpoint once
                if (checkCount > 5) {
                    checkOnce(true);
                    clearInterval(thoughtPollingId);
                }
            } finally {
                isPolling = false;
            }
        }
        
        // Function to check the regular endpoint once as fallback
        function checkOnce(isFinalAttempt = false) {
            console.log(`[Voice] Checking session status: ${sessionId}, final attempt: ${isFinalAttempt}`);
            
            fetch(`/api/check-processing/${sessionId}`)
            .then(response => response.json())
            .then(data => {
                console.log('[Voice] Session status result:', data);
                
                // Save transcript if available
                if (data.result && data.result.transcription) {
                    lastTranscript = data.result.transcription;
                } else if (data.transcription) {
                    lastTranscript = data.transcription;
                }
                
                // Check for nested result format
                if (data.result && data.result.result && typeof data.result.result === 'object') {
                    data.result = data.result.result;
                }
                
                // Check if we have a complete response
                if (data.status === 'complete' && data.result) {
                    // Format the data for display
                    let formattedResult;
                    
                    // Check if we already have a nice response object
                    if (data.result.response && typeof data.result.response === 'object') {
                        formattedResult = data.result;
                    } else {
                        formattedResult = formatApiResponseData(data.result);
                    }
                    
                    console.log('[Voice] Formatted result:', formattedResult);
                    
                    // Prevent displaying empty or error results unless this is the final attempt
                    if (!isFinalAttempt) {
                        // Check if we have a valid response
                        const hasValidResponse = 
                            (formattedResult && formattedResult.response && 
                             formattedResult.response.answer && 
                             formattedResult.response.answer !== "I apologize, but the response is taking longer than expected. Please try asking your question again.");
                        
                        if (!hasValidResponse) {
                            console.log('[Voice] Skipping invalid or incomplete response in regular check');
                            return;
                        }
                    }
                    
                    // Only clear interval and display result if this is a final attempt 
                    // or we have a valid result
                    if (isFinalAttempt || (formattedResult && formattedResult.response)) {
                    // Collapse thinking container
                    thinkingContainer.classList.add('collapsed');
                    
                    // Create a toggle button
                    const toggleButton = document.createElement('button');
                    toggleButton.className = 'toggle-thinking-btn';
                    toggleButton.innerHTML = '<i class="fas fa-lightbulb"></i> Show Thinking';
                    toggleButton.addEventListener('click', () => {
                        thinkingContainer.classList.toggle('collapsed');
                        toggleButton.innerHTML = thinkingContainer.classList.contains('collapsed') ?
                            '<i class="fas fa-lightbulb"></i> Show Thinking' :
                            '<i class="fas fa-lightbulb"></i> Hide Thinking';
                    });
                    
                    // Add toggle button to thinking header
                    if (thinkingHeader) {
                        thinkingHeader.appendChild(toggleButton);
                    }
                    
                    // Display the response
                    displayResponse(formattedResult);
                    clearInterval(thoughtPollingId);
                    }
                    return;
                }
                // Handle error responses
                else if (data.result && data.result.error) {
                    // Only show error and clear interval if this is a final attempt
                    if (isFinalAttempt) {
                    const errorResult = {
                        response: {
                            answer: `Error: ${data.result.error}`,
                            method: data.result.response || "Error during processing",
                            confidence: 0.1,
                            category: "Error"
                        },
                        transcription: lastTranscript || ""
                    };
                    
                    // Collapse thinking container
                    thinkingContainer.classList.add('collapsed');
                    
                    // Create a toggle button
                    const toggleButton = document.createElement('button');
                    toggleButton.className = 'toggle-thinking-btn';
                    toggleButton.innerHTML = '<i class="fas fa-lightbulb"></i> Show Thinking';
                    toggleButton.addEventListener('click', () => {
                        thinkingContainer.classList.toggle('collapsed');
                        toggleButton.innerHTML = thinkingContainer.classList.contains('collapsed') ?
                            '<i class="fas fa-lightbulb"></i> Show Thinking' :
                            '<i class="fas fa-lightbulb"></i> Hide Thinking';
                    });
                    
                    // Add toggle button to thinking header
                    if (thinkingHeader) {
                        thinkingHeader.appendChild(toggleButton);
                    }
                    
                    // Display the error
                    displayResponse(errorResult);
                    clearInterval(thoughtPollingId);
                    }
                    return;
                }
                
                // Handle final attempt timeout
                if (isFinalAttempt) {
                    // If we have a transcript but no response, show a fallback message
                    if (lastTranscript) {
                        const timeoutResult = {
                            response: {
                                answer: "I apologize, but the response is taking longer than expected. Please try asking your question again.",
                                method: "The system timed out while processing your query about carbon footprint calculation.",
                                confidence: 0.5,
                                category: "Error"
                            },
                            transcription: lastTranscript
                        };
                        
                        // Display timeout message
                        displayResponse(timeoutResult);
                        clearInterval(thoughtPollingId);
                    } else {
                        showErrorMessage('Response generation timed out. Please try asking again.');
                        clearInterval(thoughtPollingId);
                    }
                }
            })
            .catch(error => {
                console.error('[Voice] Error checking session status:', error);
                
                if (isFinalAttempt) {
                    showErrorMessage('Error retrieving response. Please try again.');
                    clearInterval(thoughtPollingId);
                }
            });
        }
        
        // Start the polling process
        return thoughtPollingId;
    }
    
    // Helper function to format the API response consistently
    function formatApiResponseData(apiResult) {
        console.log('[Voice] Formatting API result:', apiResult);
        
        // If response is already in the expected format, use it directly
        if (apiResult && typeof apiResult.response === 'object' && apiResult.response.answer) {
            return apiResult;
        }
        
        // If response is a string, try to parse it as JSON
        if (apiResult && typeof apiResult.response === 'string') {
            const parsedResponse = tryParseNestedJSON(apiResult.response);
            if (parsedResponse && isValidCarbonResponse(parsedResponse)) {
                return {
                    response: parsedResponse,
                    transcription: apiResult.transcription || ""
                };
            }
        }
        
        // Handle special case where result has the properties directly
        if (apiResult && typeof apiResult.answer !== 'undefined') {
            return {
                response: {
                    answer: apiResult.answer,
                    method: apiResult.method || apiResult.methodology || "Carbon Footprint Analysis",
                    confidence: apiResult.confidence || 0.8,
                    category: apiResult.category || "Analysis"
                },
                transcription: apiResult.transcription || ""
            };
        }
        
        // If we have an answer field nested under result
        if (apiResult && apiResult.result && typeof apiResult.result.answer !== 'undefined') {
            return {
                response: {
                    answer: apiResult.result.answer,
                    method: apiResult.result.method || apiResult.result.methodology || "Carbon Footprint Analysis",
                    confidence: apiResult.result.confidence || 0.8,
                    category: apiResult.result.category || "Analysis"
                },
                transcription: apiResult.transcription || ""
            };
        }
        
        // Handle case where response is direct text
        if (apiResult && typeof apiResult === 'string') {
            return {
                response: {
                    answer: apiResult,
                    method: "Direct Response",
                    confidence: 0.7,
                    category: "Analysis"
                },
                transcription: ""
            };
        }
        
        // Last resort - try to extract any text content
        if (apiResult) {
            // Look for any property that might contain the answer
            const possibleAnswerFields = ['response', 'text', 'content', 'message', 'output'];
            for (const field of possibleAnswerFields) {
                if (apiResult[field] && typeof apiResult[field] === 'string' && apiResult[field].length > 10) {
                    return {
                        response: {
                            answer: apiResult[field],
                            method: "Extracted Response",
                            confidence: 0.6,
                            category: "Analysis"
                        },
                        transcription: apiResult.transcription || ""
                    };
                }
            }
        }
        
        // Default format - return as is
        return apiResult;
    }
    
    // Display the final response
    function displayResponse(result) {
        console.log('[Voice] Displaying response:', result);
        
        // Remove any status indicator
        removeStatusIndicator();
        
        // Find the most recent thinking container
        const thinkingContainer = document.querySelector('.thinking-container:last-child');
        
        // Check for nested result structure (result.result pattern)
        if (result && result.result && typeof result.result === 'object') {
            console.log('[Voice] Detected nested result structure, extracting result.result');
            result = {
                response: result.result,
                transcription: result.transcription || ""
            };
        }
        
        // Display the bot's response
        if (result && (result.response || typeof result === 'string')) {
            if (typeof window.addBotMessage === 'function') {
                console.log('[Voice] Using global addBotMessage function from dashboard.js');
                
                // Format the result to match what dashboard.js expects
                let formattedResult;
                
                // Handle different response formats
                if (typeof result === 'string') {
                    // Simple string result
                    formattedResult = result;
                } else if (typeof result.response === 'string') {
                    try {
                        // Try to parse the string as JSON
                        const parsedResponse = tryParseNestedJSON(result.response);
                        if (parsedResponse && isValidCarbonResponse(parsedResponse)) {
                            formattedResult = parsedResponse;
                        } else {
                            // If not valid JSON, use as plain text
                            formattedResult = result.response;
                        }
                    } catch (e) {
                        console.error('[Voice] Error parsing response string:', e);
                        formattedResult = result.response;
                    }
                } else if (typeof result.response === 'object') {
                    // If it's already an object, use it directly
                    formattedResult = result.response;
                } else {
                    // Fallback for any other type
                    formattedResult = String(result.response || result);
                }
                
                console.log('[Voice] Formatted result for addBotMessage:', formattedResult);
                
                // Pass the formatted result to the global addBotMessage function
                window.addBotMessage(formattedResult);
            } else {
                console.log('[Voice] Using local display method (addBotMessage not available)');
                // Create message container
                const messageContainer = document.createElement('div');
                messageContainer.className = 'message bot';
                
                // Check if result is a string or an object
                if (typeof result === 'string') {
                    // Try to parse string as JSON first
                    const parsedResponse = tryParseNestedJSON(result);
                    
                    if (parsedResponse && isValidCarbonResponse(parsedResponse)) {
                        displayStructuredResponse(messageContainer, parsedResponse);
                    } else {
                        // Simple string response
                        messageContainer.textContent = result;
                    }
                } else if (result.response) {
                    // Result with response field
                    if (typeof result.response === 'string') {
                        // Try to parse string response as JSON
                        const parsedResponse = tryParseNestedJSON(result.response);
                        
                        if (parsedResponse && isValidCarbonResponse(parsedResponse)) {
                            displayStructuredResponse(messageContainer, parsedResponse);
                        } else {
                            // Simple string response
                            messageContainer.textContent = result.response;
                        }
                    } else if (typeof result.response === 'object') {
                        // Object response
                        displayStructuredResponse(messageContainer, result.response);
                    } else {
                        // Fallback for any other type
                        messageContainer.textContent = String(result.response);
                    }
                } else {
                    // No response field but might have direct properties like "answer"
                    if (typeof result.answer !== 'undefined') {
                        displayStructuredResponse(messageContainer, result);
                    } else {
                        // Last resort - stringify the object
                        messageContainer.textContent = JSON.stringify(result);
                    }
                }
                
                // Add to conversation and scroll
                conversationContainer.appendChild(messageContainer);
                ensureScrolling();
            }
            
            // If we have a thinking container, collapse it and add a toggle button
            if (thinkingContainer) {
                // Don't collapse the thinking container, leave it fully visible
                // thinkingContainer.classList.add('collapsed');
                
                // Check if toggle button already exists
                if (!thinkingContainer.querySelector('.toggle-thinking-btn')) {
                    // Create a toggle button
                    const toggleButton = document.createElement('button');
                    toggleButton.className = 'toggle-thinking-btn';
                    toggleButton.innerHTML = '<i class="fas fa-lightbulb"></i> Hide Thinking';
                    toggleButton.addEventListener('click', () => {
                        thinkingContainer.classList.toggle('collapsed');
                        toggleButton.innerHTML = thinkingContainer.classList.contains('collapsed') ?
                            '<i class="fas fa-lightbulb"></i> Show Thinking' :
                            '<i class="fas fa-lightbulb"></i> Hide Thinking';
                        
                        // After toggling, ensure scrolling
                        ensureScrolling();
                    });
                    
                    // Add toggle button to thinking header
                    const thinkingHeader = thinkingContainer.querySelector('.thinking-header');
                    if (thinkingHeader) {
                        thinkingHeader.appendChild(toggleButton);
                    }
                }
                
                // Ensure conversation is scrolled to the latest content
                ensureScrolling();
            }
            
            // Clean up temporary files
            cleanupTemporaryFiles();
        }
    }
    
    // Clean up temporary files created during voice processing
    function cleanupTemporaryFiles() {
        console.log('[Voice] Cleaning up temporary files');
        
        // Clean up transcript file
        fetch('/api/cleanup-transcript', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }).catch(error => {
            console.error('[Voice] Error cleaning up transcript:', error);
        });
        
        // Reset the global transcript variable
        fetch('/api/reset-transcript', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }).catch(error => {
            console.error('[Voice] Error resetting transcript:', error);
        });
        
        // Clean up thoughts
        fetch('/api/cleanup-thoughts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }).catch(error => {
            console.error('[Voice] Error cleaning up thoughts:', error);
        });
    }
    
    // Helper function to display structured response 
    function displayStructuredResponse(result) {
        console.log('[Voice] Displaying structured response:', result);
        
        // Handle different response formats
        let content, carbonValue = null, category = null;
        
        if (typeof result === 'string') {
            // Simple string response
            content = result;
        } else if (result && typeof result === 'object') {
            // Extract content based on available fields
            if (result.result) {
                if (typeof result.result === 'string') {
                    content = result.result;
                } else if (typeof result.result === 'object') {
                    content = result.result.answer || result.result.final_answer || JSON.stringify(result.result);
                }
            } else if (result.answer || result.final_answer) {
                content = result.answer || result.final_answer;
            } else {
                content = JSON.stringify(result);
            }
            
            // Extract carbon value and category if available
            // Check in different possible locations based on the structure
            if (result.result && result.result.estimated_footprints) {
                const footprints = result.result.estimated_footprints;
                if (footprints.length > 0) {
                    carbonValue = footprints[0].value;
                    category = footprints[0].category || footprints[0].entity;
                }
            } else if (result.estimated_footprints) {
                const footprints = result.estimated_footprints;
                if (footprints.length > 0) {
                    carbonValue = footprints[0].value;
                    category = footprints[0].category || footprints[0].entity;
                }
            } else if (result.result && result.result.carbon_value) {
                carbonValue = result.result.carbon_value;
                category = result.result.category;
            } else if (result.carbon_value) {
                carbonValue = result.carbon_value;
                category = result.category;
            }
        } else {
            content = "Could not parse response";
        }
        
        // Clean up the content if needed
        if (!content) {
            content = "No content available in the response";
        }
        
        // Add the response to the chat
        addBotMessage(content);
        
        // Add Track Impact button if we have a carbon value
        if (carbonValue !== null && category !== null) {
            console.log('[Voice] Adding Track Impact button for:', carbonValue, category);
            
            // Create a container for the button
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'track-button-container';
            
            // Create the button
            const trackButton = document.createElement('button');
            trackButton.className = 'btn btn-primary track-query-btn';
            trackButton.innerHTML = '<i class="fas fa-chart-line"></i> Track Impact';
            
            // Set up the event listener
            trackButton.addEventListener('click', function(event) {
                trackQuery(content, category, carbonValue, event);
            });
            
            // Add button to container
            buttonContainer.appendChild(trackButton);
            
            // Add the button container to the chat
            const messagesContainer = document.getElementById('messages');
            if (messagesContainer) {
                messagesContainer.appendChild(buttonContainer);
            } else {
                console.error('[Voice] Could not find messages container');
            }
        }
        
        // Scroll to the bottom of the messages container
        const messagesContainer = document.getElementById('messages');
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        // Hide spinner and enable record button
        hideSpinner();
        enableRecordButton();
    }
    
    // Helper function to check if an object looks like a valid carbon response (from dashboard.js)
    function isValidCarbonResponse(obj) {
        // Check if it's a full response format
        if (obj && typeof obj === 'object') {
            // New response format that should contain answer field
            if (typeof obj.answer !== 'undefined') {
                return true;
            }
            
            // Alternative format with result field
            if (obj.result && typeof obj.result === 'object') {
                return typeof obj.result.answer !== 'undefined';
            }
            
            // Check for complete standard format
            if (typeof obj.answer !== 'undefined' && 
                (typeof obj.method !== 'undefined' || typeof obj.methodology !== 'undefined') && 
                typeof obj.confidence !== 'undefined' && 
                typeof obj.category !== 'undefined') {
                return true;
            }
            
            // Simplified response format
            if (typeof obj.response !== 'undefined') {
                // If response is an object, it should have an answer
                if (typeof obj.response === 'object') {
                    return typeof obj.response.answer !== 'undefined';
                }
                // If response is a string, it should be non-empty
                if (typeof obj.response === 'string' && obj.response.trim().length > 0) {
                return true;
                }
            }
            
            // Check for nested response.response patterns (common in some API responses)
            if (obj.response && obj.response.response) {
                if (typeof obj.response.response === 'object') {
                    return typeof obj.response.response.answer !== 'undefined';
                }
                if (typeof obj.response.response === 'string' && obj.response.response.trim().length > 0) {
                    return true;
                }
            }
        }
        return false;
    }
    
    // Helper function for parsing nested JSON (from dashboard.js)
    function tryParseNestedJSON(jsonString) {
        if (typeof jsonString !== 'string') return null;
        
        // Skip if not likely to be JSON
        if (!jsonString.trim().startsWith('{') && !jsonString.trim().startsWith("{'") && 
            !jsonString.trim().startsWith('[') && !jsonString.trim().startsWith("['")) {
            return null;
        }
        
        try {
            // Try standard JSON format first
            if ((jsonString.trim().startsWith('{') && jsonString.trim().endsWith('}')) ||
                (jsonString.trim().startsWith('[') && jsonString.trim().endsWith(']'))) {
                return JSON.parse(jsonString);
            }
            
            // Try parsing Python-style dict with single quotes
            if ((jsonString.trim().startsWith("{'") && jsonString.trim().endsWith("'}")) ||
                (jsonString.trim().startsWith("['") && jsonString.trim().endsWith("']"))) {
                const jsonStr = jsonString
                    .replace(/'/g, '"')  // Replace single quotes with double quotes
                    .replace(/None/g, 'null')  // Replace None with null
                    .replace(/True/g, 'true')  // Replace True with true
                    .replace(/False/g, 'false'); // Replace False with false
                
                return JSON.parse(jsonStr);
            }
            
            // Search for JSON-like content within the string (for when JSON is embedded in text)
            const jsonRegex = /{[\s\S]*?}/g;
            const match = jsonString.match(jsonRegex);
            if (match && match[0]) {
                try {
                    const extracted = match[0];
                    return JSON.parse(extracted);
                } catch (e) {
                    // If nested extraction fails, just continue with other attempts
                    console.warn('[Voice] Failed to extract nested JSON:', e);
                }
            }
        } catch (e) {
            console.error("[Voice] Failed to parse nested JSON:", e);
        }
        
        return null;
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
            ensureScrolling();
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
            ensureScrolling();
        }
    }
}

// Add the trackQuery function if it doesn't exist already in this context
function trackQuery(answer, category, carbonValue, event) {
    console.log('[Voice] Tracking query with carbonValue:', carbonValue, 'category:', category);
    
    // Use the global trackQuery if it exists
    if (typeof window.trackQuery === 'function') {
        console.log('[Voice] Using global trackQuery function from dashboard.js');
        window.trackQuery(answer, category, carbonValue, event);
        return;
    }
    
    // Fallback implementation if global trackQuery is not available
    console.log('[Voice] Using local trackQuery implementation');
    
    // Get target button from event or find it
    const targetButton = event ? event.currentTarget : document.querySelector('.track-query-btn');
    if (!targetButton) {
        console.error('[Voice] Could not find track button');
        return;
    }
    
    // Disable the button to prevent multiple submissions
    targetButton.disabled = true;
    targetButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Tracking...';
    
    // Send tracking request
    fetch('/api/track-query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: answer,
            category: category,
            carbonValue: carbonValue
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('[Voice] Query tracked successfully:', data);
        
        // Update button to show success
        targetButton.innerHTML = '<i class="fas fa-check"></i> Tracked';
        targetButton.classList.add('tracked');
        
        // Show tracking confirmation toast if function exists
        if (typeof window.showToast === 'function') {
            window.showToast('Query tracked successfully!', 'success');
        } else {
            // Simple alert fallback
            alert('Carbon impact tracked successfully!');
        }
    })
    .catch(error => {
        console.error('[Voice] Error tracking query:', error);
        
        // Reset button
        targetButton.disabled = false;
        targetButton.innerHTML = '<i class="fas fa-chart-line"></i> Track Impact';
        
        // Show error toast if function exists
        if (typeof window.showToast === 'function') {
            window.showToast('Failed to track query. Please try again.', 'error');
        } else {
            // Simple alert fallback
            alert('Failed to track carbon impact. Please try again.');
        }
    });
}

// Export to make functions available to other modules
window.VoiceHandler = {
    trackQuery: trackQuery
};

console.log('[Voice] Voice handler module loaded');

// --- Streaming agent step logic for voice mode ---
function streamAgentStepsVoice(query) {
    console.log('[Voice] Starting agent step streaming for voice query');
    const conversationContainer = document.getElementById('conversation-container');
    const requestId = `query_${Date.now()}`;
    
    // Remove any previous status indicators
    removeStatusIndicator();

    // Create a new thinking/status container
    const thinkingContainer = document.createElement('div');
    thinkingContainer.className = 'thinking-container';
    
    // Create header for thinking container
    const thinkingHeader = document.createElement('div');
    thinkingHeader.className = 'thinking-header';
    thinkingHeader.innerHTML = '<i class="fas fa-robot"></i> Thinking...';
    thinkingContainer.appendChild(thinkingHeader);
    
    // Create thoughts area with no max-height limit
    const thoughtsArea = document.createElement('div');
    thoughtsArea.className = 'thoughts-area';
    thoughtsArea.style.maxHeight = 'none'; // Remove max-height limit
    thinkingContainer.appendChild(thoughtsArea);
    
    // Create initial thought element
    const initialThought = document.createElement('div');
    initialThought.className = 'thought start';
    initialThought.innerHTML = `
        <span class="thought-icon"><i class="fas fa-play"></i></span>
        <span class="thought-content">Starting analysis of your voice query: "${query}"</span>
    `;
    thoughtsArea.appendChild(initialThought);
    
    // Add container to conversation
    conversationContainer.appendChild(thinkingContainer);
    
    // Ensure scrollability
    ensureScrolling();

    // Start SSE connection with request ID for tracking
    console.log(`[Voice] Creating SSE connection with request ID: ${requestId}`);
    const evtSource = new EventSource(`/api/stream-agent-step?query=${encodeURIComponent(query)}&request_id=${requestId}`);
    
    // Handle incoming messages
    evtSource.onmessage = function(event) {
        try {
            // Skip keep-alive messages
            if (event.data.trim() === '') return;
            
            const data = JSON.parse(event.data);
            console.log('[Voice] Received agent step update:', data);
            
            // Check for completion
            if (data.agent === 'done') {
                console.log('[Voice] Agent processing complete');
                thinkingHeader.innerHTML = '<i class="fas fa-check-circle"></i> Analysis Complete!';
                
                // Add a final completion thought
                const completionThought = document.createElement('div');
                completionThought.className = 'thought complete';
                completionThought.innerHTML = `
                    <span class="thought-icon"><i class="fas fa-check-circle"></i></span>
                    <span class="thought-content">Analysis complete! Preparing final response...</span>
                `;
                thoughtsArea.appendChild(completionThought);
                
                // Ensure scrollability
                ensureScrolling();
                
                // Close the connection
                evtSource.close();
                
                // Add a toggle button
                const toggleButton = document.createElement('button');
                toggleButton.className = 'toggle-thinking-btn';
                toggleButton.innerHTML = '<i class="fas fa-lightbulb"></i> Hide Thinking';
                toggleButton.addEventListener('click', () => {
                    thinkingContainer.classList.toggle('collapsed');
                    toggleButton.innerHTML = thinkingContainer.classList.contains('collapsed') ?
                        '<i class="fas fa-lightbulb"></i> Show Thinking' :
                        '<i class="fas fa-lightbulb"></i> Hide Thinking';
                    
                    // After toggling, ensure scrolling
                    ensureScrolling();
                });
                
                // Add toggle button to thinking header
                if (thinkingHeader) {
                    thinkingHeader.appendChild(toggleButton);
                }
                
                // Don't collapse thinking container by default
                // thinkingContainer.classList.add('collapsed');
                
                // Ensure scrolling one more time
                ensureScrolling();
                
                return;
            }
            
            // Get the appropriate message for this agent
            const msg = window.getAgentMessage ? 
                window.getAgentMessage(data.agent) : 
                (data.message || `Processing with ${data.agent}...`);
            
            // Create a new thought for this agent step
            const agentThought = document.createElement('div');
            agentThought.className = 'thought agent_step';
            agentThought.innerHTML = `
                <span class="thought-icon"><i class="fas fa-cog fa-spin"></i></span>
                <span class="thought-content">${msg}</span>
            `;
            
            // Add the new thought
            thoughtsArea.appendChild(agentThought);
            
            // Ensure scrollability
            ensureScrolling();
            
        } catch (e) {
            console.error('[Voice] Error processing SSE message:', e, event.data);
        }
    };
    
    // Handle connection open
    evtSource.onopen = function() {
        console.log('[Voice] SSE connection opened');
    };
    
    // Handle errors
    evtSource.onerror = function(err) {
        console.error('[Voice] SSE connection error:', err);
        
        // Add an error thought
        const errorThought = document.createElement('div');
        errorThought.className = 'thought error';
        errorThought.innerHTML = `
            <span class="thought-icon"><i class="fas fa-exclamation-triangle"></i></span>
            <span class="thought-content">Connection error. The system may still be processing your query.</span>
        `;
        thoughtsArea.appendChild(errorThought);
        
        // Ensure scrollability
        ensureScrolling();
        
        // Close the connection
        evtSource.close();
    };
}

// Update the ensureScrolling function to not rely on a global conversationContainer variable
function ensureScrolling() {
    // Get the conversation container element directly instead of relying on global scope
    const conversationContainer = document.getElementById('conversation-container');
    
    // Ensure the main conversation container is scrolled down
    if (conversationContainer) {
        // Force layout recalculation to get accurate scrollHeight
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
        
        // In some browsers, an immediate second scroll ensures animation completes
        setTimeout(() => {
            conversationContainer.scrollTop = conversationContainer.scrollHeight;
        }, 50);
    }
    
    // Also ensure any thoughts areas are scrolled
    const thoughtsAreas = document.querySelectorAll('.thoughts-area');
    thoughtsAreas.forEach(area => {
        if (area) {
            area.scrollTop = area.scrollHeight;
        }
    });
}
