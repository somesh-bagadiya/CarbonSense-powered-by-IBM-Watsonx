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
        
        // Create thinking container similar to dashboard.js
        const thinkingContainer = document.createElement('div');
        thinkingContainer.className = 'thinking-container';
        
        // Create header for thinking container
        const thinkingHeader = document.createElement('div');
        thinkingHeader.className = 'thinking-header';
        thinkingHeader.innerHTML = '<i class="fas fa-microphone"></i> Recording...';
        thinkingContainer.appendChild(thinkingHeader);
        
        // Create thoughts area
        const thoughtsArea = document.createElement('div');
        thoughtsArea.className = 'thoughts-area';
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
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
        
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
                
                // Scroll to bottom
                thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
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
    
    // Fetch transcription after delay
    function fetchTranscription() {
        console.log('[Voice] Fetching transcription after delay');
        
        // Implement polling for the transcript file
        let checkCount = 0;
        const maxChecks = 30; // Maximum number of checks
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
                    
                    // Start the result generation phase
                    startResultGeneration();
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
                            
                            // Start the result generation phase
                            startResultGeneration();
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
                                startResultGeneration();
                            }
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
        
        // Find the thinking container and update it
        const thinkingContainer = document.querySelector('.thinking-container');
        
        if (thinkingContainer) {
            // Update the header
            const header = thinkingContainer.querySelector('.thinking-header');
            if (header) {
                header.innerHTML = '<i class="fas fa-robot"></i> Generating Response...';
            }
            
            // Add a new thought about generation
            const thoughtsArea = thinkingContainer.querySelector('.thoughts-area');
            if (thoughtsArea) {
                const generationThought = document.createElement('div');
                generationThought.className = 'thought action';
                generationThought.innerHTML = `
                    <span class="thought-icon"><i class="fas fa-cog fa-spin"></i></span>
                    <span class="thought-content">Analyzing transcript and generating a thoughtful response...</span>
                `;
                thoughtsArea.appendChild(generationThought);
                thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
            }
        } else {
            // Fallback to old method
        showStatusIndicator('generating');
        }
        
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
        const maxChecks = 40; // Maximum number of checks (20 * 3s = 60 seconds max)
        let lastTranscript = null;
        let thoughtPollingId = null; // For thought tracking
        let lastSeenThoughtIndex = -1; // Track which thoughts we've seen
        let isPolling = false; // Prevent concurrent polling

        // Create thinking container similar to dashboard.js
        const thinkingContainer = document.createElement('div');
        thinkingContainer.className = 'thinking-container';
        
        // Create header for thinking container
        const thinkingHeader = document.createElement('div');
        thinkingHeader.className = 'thinking-header';
        thinkingHeader.innerHTML = '<i class="fas fa-robot"></i> Thinking...';
        thinkingContainer.appendChild(thinkingHeader);
        
        // Create thoughts area
        const thoughtsArea = document.createElement('div');
        thoughtsArea.className = 'thoughts-area';
        thinkingContainer.appendChild(thoughtsArea);
        
        // Create a single reusable thought element
        const thoughtElement = document.createElement('div');
        thoughtElement.className = 'thought';
        thoughtElement.innerHTML = `
            <span class="thought-icon"><i class="fas fa-brain"></i></span>
            <span class="thought-content">Initializing analysis...</span>
        `;
        thoughtsArea.appendChild(thoughtElement);
        
        // Add to conversation
        conversationContainer.appendChild(thinkingContainer);
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
        
        // Remove status indicator since we're using the thinking container
        removeStatusIndicator();
        
        // Start thought polling immediately
        startThoughtPolling();
        
        // Also start the regular result checking as a fallback
        checkOnce();
        resultCheckTimer = setInterval(checkOnce, 3000);
        
        // Function to start thought polling
        function startThoughtPolling() {
            console.log('[Voice] Starting independent thought polling');
            thoughtPollingId = {};  // Use an object as a non-null identifier
            pollForThoughts();
        }
        
        // Function to poll for thoughts independently
        async function pollForThoughts() {
            // Prevent multiple concurrent polling
            if (isPolling) return;
            
            isPolling = true;
            
            try {
                console.log('[Voice] Polling for thoughts directly');
                
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
                        
                        // Get the latest thought (we're only displaying the latest one)
                        const latestThought = newThoughts[newThoughts.length - 1];
                    
                        // Update last seen thought index
                        lastSeenThoughtIndex = thoughtsData.thoughts.length - 1;
                        
                        // Update the thought element with the latest thought
                        updateThoughtElement(thoughtElement, latestThought);
                        
                        // Scroll thoughts area down to show updated thought
                        thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                    }
                    
                    // Check status for completion or error
                    if (thoughtsData.status === 'COMPLETE' || thoughtsData.status === 'complete') {
                        console.log('[Voice] Thought process complete');
                        
                        // Add final thought
                        updateThoughtElement(thoughtElement, {
                            content: 'Analysis complete!',
                            type: 'complete'
                        });
                        
                        // Retrieve the final answer from a completion thought if available
                        const completionThoughts = thoughtsData.thoughts.filter(t => t.type === 'completion');
                        if (completionThoughts.length > 0) {
                            // Get the most recent completion thought
                            const latestCompletion = completionThoughts.sort((a, b) => b.timestamp - a.timestamp)[0];
                            
                            // Create a result object with the completion thought content
                            const result = {
                                response: latestCompletion.content,
                                transcription: lastTranscript || '',
                                confidence: 0.8
                            };
                            
                            // Display the response and stop polling
                            clearInterval(resultCheckTimer);
                            resultCheckTimer = null;
                            stopPolling();
                            
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
                            thinkingHeader.appendChild(toggleButton);
                            
                            // Display the response
                            displayResponse(result);
                            return;
                        }
                        
                        // Scroll thoughts area down
                        thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                    } else if (thoughtsData.status === 'ERROR' || thoughtsData.status === 'error') {
                        console.log('[Voice] Thought process error');
                        
                        // Update with error thought
                        updateThoughtElement(thoughtElement, {
                            content: 'Error in thought processing',
                            type: 'error'
                        });
                        
                        // Find error thoughts to display a more specific message
                        const errorThoughts = thoughtsData.thoughts.filter(t => t.type === 'error');
                        if (errorThoughts.length > 0) {
                            // Get the most recent error thought
                            const latestError = errorThoughts.sort((a, b) => b.timestamp - a.timestamp)[0];
                            
                            // Create an error result object
                            const errorResult = {
                                response: latestError.content || 'An error occurred during processing.',
                                transcription: lastTranscript || '',
                                confidence: 0.5,
                                error: 'Processing error'
                            };
                            
                            // Stop checking and display the error
                            clearInterval(resultCheckTimer);
                            resultCheckTimer = null;
                            stopPolling();
                            displayResponse(errorResult);
                            return;
                        }
                        
                        // Scroll thoughts area down
                        thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                    }
                } else {
                    console.log('[Voice] No thoughts available yet');
                }
            } catch (error) {
                console.error('[Voice] Error fetching thoughts:', error);
            } finally {
                isPolling = false;
                
                // Continue polling with a delay regardless of backend response
                if (thoughtPollingId !== null) {
                    // Use exponential backoff for polling frequency
                    const delayMs = Math.min(1000 * Math.pow(1.2, Math.min(checkCount, 10)), 5000);
                    console.log(`[Voice] Next thought poll in ${delayMs/1000} seconds`);
                    
                    setTimeout(() => {
                        if (thoughtPollingId !== null) {  // Check again in case stopped during timeout
                            pollForThoughts();
                        }
                    }, delayMs);
                }
            }
        }
        
        // Function to update the thought element, same as in dashboard.js
        function updateThoughtElement(element, thought) {
            // Get thought type or default to 'thought'
            const thoughtType = thought.type || 'thought';
            
            // Only change the class if the type has changed
            if (element.className !== `thought ${thoughtType}`) {
                // Update element class
                element.className = `thought ${thoughtType}`;
                
                // Update icon based on type
                let icon = '';
                switch(thoughtType) {
                    case 'thought':
                    case 'thinking':
                        icon = '<i class="fas fa-brain"></i>';
                        break;
                    case 'action':
                        icon = '<i class="fas fa-cog fa-spin"></i>';
                        break;
                    case 'error':
                        icon = '<i class="fas fa-exclamation-triangle"></i>';
                        break;
                    case 'complete':
                    case 'completion':
                        icon = '<i class="fas fa-check-circle"></i>';
                        break;
                    default:
                        icon = '<i class="fas fa-comment"></i>';
                }
                
                // Update the icon
                const iconElement = element.querySelector('.thought-icon');
                if (iconElement) {
                    iconElement.innerHTML = icon;
                }
            }
            
            // Ensure thought.content exists before using it
            const content = thought.content || 'No content available';
            
            // Update the content
            const contentElement = element.querySelector('.thought-content');
            if (contentElement) {
                contentElement.textContent = content;
            }
            
            console.log(`[Voice] Updated thought: ${content} (${thoughtType})`);
        }
        
        function stopPolling() {
            console.log('[Voice] Stopping thought polling');
            
            if (thoughtPollingId !== null) {
                thoughtPollingId = null;
                // No need to clean up thoughts as we want to keep them visible in the UI
            }
        }

        // Regular result checking as fallback
        function checkOnce() {
            checkCount++;
            console.log(`[Voice] Result check ${checkCount}/${maxChecks}`);
            
            // Stop checking after maximum attempts
            if (checkCount >= maxChecks) {
                clearInterval(resultCheckTimer);
                resultCheckTimer = null;
                stopPolling();
                
                // If we have a transcript but no complete response, show partial result
                if (lastTranscript) {
                    const fallbackResult = {
                        response: `I processed your query: "${lastTranscript}" but couldn't generate a complete response in time.`,
                        transcription: lastTranscript,
                        confidence: 0.5
                    };
                    
                    displayResponse(fallbackResult);
                } else {
                showErrorMessage('Response generation timed out. Please try asking again.');
                }
                return;
            }
            
            fetch(`/api/check-processing/${sessionId}`)
            .then(response => response.json())
            .then(data => {
                console.log('[Voice] Result check response:', data);
                
                // Save the transcript we're seeing
                if (data.result && data.result.transcription) {
                    lastTranscript = data.result.transcription;
                } else if (data.transcription) {
                    lastTranscript = data.transcription;
                }
                
                // Check if we have a full response with response field
                if (data.status === 'complete' && data.result && data.result.response) {
                    // We have the response, display it
                    clearInterval(resultCheckTimer);
                    resultCheckTimer = null;
                    stopPolling();
                    
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
                    thinkingHeader.appendChild(toggleButton);
                    
                    displayResponse(data.result);
                }
                // Check for errors
                else if (data.result && data.result.error) {
                    console.log('[Voice] Detected processing error');
                    clearInterval(resultCheckTimer);
                    resultCheckTimer = null;
                    stopPolling();
                    
                    // Try to reset the system if it's a CrewAI error
                    if (data.result.error === "CrewAI processing error") {
                        fetch('/api/reset', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                    .then(response => response.json())
                    .then(resetData => {
                            console.log('[Voice] System reset result:', resetData);
                    })
                    .catch(error => {
                            console.error('[Voice] Error resetting system:', error);
                        });
                    }
                    
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
                    thinkingHeader.appendChild(toggleButton);
                    
                    // Display the error response
                    displayResponse(data.result);
                }
            })
            .catch(error => {
                console.error('[Voice] Error checking for results:', error);
                
                // Stop checking after 3 consecutive errors
                if (checkCount > 3) {
                    clearInterval(resultCheckTimer);
                    resultCheckTimer = null;
                    stopPolling();
                    showErrorMessage('Error retrieving response. Please try again.');
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
                
                // Just pass the result directly to addBotMessage which will handle formatting
                window.addBotMessage(result);
            } else {
                console.log('[Voice] Using fallback display method');
                // Create message container
                const messageContainer = document.createElement('div');
                messageContainer.className = 'message bot';
                
                // Check if result is a string or an object
                if (typeof result.response === 'string') {
                    // Parse result.response for nested JSON
                    let parsedResponse = null;
                    try {
                        // Try standard JSON format first
                        if (result.response.trim().startsWith('{') && result.response.trim().endsWith('}')) {
                            parsedResponse = JSON.parse(result.response);
                        }
                        // Try parsing Python-style dict with single quotes
                        else if (result.response.trim().startsWith("{'") && result.response.trim().endsWith("'}")) {
                            const jsonStr = result.response
                                .replace(/'/g, '"')  // Replace single quotes with double quotes
                                .replace(/None/g, 'null')  // Replace None with null
                                .replace(/True/g, 'true')  // Replace True with true
                                .replace(/False/g, 'false'); // Replace False with false
                            
                            parsedResponse = JSON.parse(jsonStr);
                        }
                    } catch (e) {
                        console.error("[Voice] Failed to parse nested JSON:", e);
                    }
                    
                    // If we have a parsed response and it looks like a carbon response, use it
                    if (parsedResponse && 
                        typeof parsedResponse === 'object' && 
                        typeof parsedResponse.answer !== 'undefined' && 
                        typeof parsedResponse.method !== 'undefined' && 
                        typeof parsedResponse.confidence !== 'undefined' && 
                        typeof parsedResponse.category !== 'undefined') {
                        
                        // Create structured response elements
                        
                        // Create answer paragraph with label
                        const answerSection = document.createElement('div');
                        answerSection.className = 'response-section';
                        
                        const answerLabel = document.createElement('strong');
                        answerLabel.textContent = 'Answer: ';
                        
                        const answerText = document.createElement('span');
                        answerText.textContent = parsedResponse.answer;
                        
                        answerSection.appendChild(answerLabel);
                        answerSection.appendChild(answerText);
                        messageContainer.appendChild(answerSection);
                        
                        // Method/Methodology section
                        if (parsedResponse.method) {
                            const methodSection = document.createElement('div');
                            methodSection.className = 'response-section';
                            
                            const methodLabel = document.createElement('strong');
                            methodLabel.textContent = 'Methodology: ';
                            
                            const methodText = document.createElement('span');
                            methodText.textContent = parsedResponse.method;
                            
                            methodSection.appendChild(methodLabel);
                            methodSection.appendChild(methodText);
                            messageContainer.appendChild(methodSection);
                        }
                        
                        // Confidence section
                        if (parsedResponse.confidence !== undefined) {
                            const confidenceSection = document.createElement('div');
                            confidenceSection.className = 'response-section';
                            
                            const confidenceLabel = document.createElement('strong');
                            confidenceLabel.textContent = 'Confidence: ';
                            
                            const confidenceText = document.createElement('span');
                            confidenceText.textContent = parsedResponse.confidence;
                            
                            confidenceSection.appendChild(confidenceLabel);
                            confidenceSection.appendChild(confidenceText);
                            messageContainer.appendChild(confidenceSection);
                        }
                        
                        // Category section
                        if (parsedResponse.category && parsedResponse.category !== 'Unknown') {
                            const categorySection = document.createElement('div');
                            categorySection.className = 'response-section';
                            
                            const categoryLabel = document.createElement('strong');
                            categoryLabel.textContent = 'Category: ';
                            
                            const categoryText = document.createElement('span');
                            categoryText.textContent = parsedResponse.category;
                            
                            categorySection.appendChild(categoryLabel);
                            categorySection.appendChild(categoryText);
                            messageContainer.appendChild(categorySection);
                        }
                        
                        // Add Track Impact button if applicable
                        if (parsedResponse.category && parsedResponse.answer) {
                            // Try to extract numeric value from the answer
                            const carbonPatterns = [
                                /(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\s*(?:of)?\s*(?:CO2e?|carbon dioxide equivalent)/i,
                                /(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of)?\s*(?:CO2e?|carbon dioxide equivalent)/i,
                                /(\d+(?:\.\d+)?)\s*(?:tons?|tonnes?)\s*(?:of)?\s*(?:CO2e?|carbon dioxide equivalent)/i,
                                /(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)/i,  // Less specific fallback
                                /(\d+(?:\.\d+)?)\s*(?:g|grams?)/i,      // Less specific fallback
                                /(\d+(?:\.\d+)?)\s*(?:CO2e?)/i,         // Just the number with CO2
                                /carbon footprint of (\d+(?:\.\d+)?)/i,  // Phrases like "carbon footprint of X"
                                /(\d+(?:\.\d+)?)/                       // Last resort: just find any number
                            ];
                            
                            let carbonValue = null;
                            let unit = 'kg';  // Default unit
                            
                            // Try each pattern until we find a match
                            for (const pattern of carbonPatterns) {
                                const match = parsedResponse.answer.match(pattern);
                                if (match) {
                                    carbonValue = parseFloat(match[1]);
                                    
                                    // Adjust for different units
                                    if (pattern.source.includes('grams?') && !pattern.source.includes('kg')) {
                                        carbonValue = carbonValue / 1000;  // Convert grams to kg
                                        unit = 'kg (converted from g)';
                                    } else if (pattern.source.includes('tons?|tonnes?')) {
                                        carbonValue = carbonValue * 1000;  // Convert tons to kg
                                        unit = 'kg (converted from tons)';
                                    }
                                    
                                    console.log(`[Voice] Extracted carbon value: ${carbonValue} ${unit} using pattern: ${pattern}`);
                                    break;
                                }
                            }
                            
                            if (carbonValue !== null) {
                                const trackButton = document.createElement('button');
                                trackButton.className = 'track-query-btn';
                                trackButton.innerHTML = '<i class="fas fa-chart-line"></i> Track Impact';
                                
                                // Add click handler with proper event passing
                                trackButton.addEventListener('click', function(e) {
                                    // Call the trackQuery function with the extracted value
                                    trackQuery(parsedResponse.answer, parsedResponse.category, carbonValue, e);
                                });
                                
                                messageContainer.appendChild(trackButton);
                            }
                        }
            } else {
                        // Simple string response
                        messageContainer.textContent = result.response;
                    }
                } else if (typeof result.response === 'object') {
                    // If it's already an object with answer field, create structured response
                    if (result.response.answer) {
                        // Create answer paragraph with label
                        const answerSection = document.createElement('div');
                        answerSection.className = 'response-section';
                        
                        const answerLabel = document.createElement('strong');
                        answerLabel.textContent = 'Answer: ';
                        
                        const answerText = document.createElement('span');
                        answerText.textContent = result.response.answer;
                        
                        answerSection.appendChild(answerLabel);
                        answerSection.appendChild(answerText);
                        messageContainer.appendChild(answerSection);
                        
                        // Method/Methodology section
                        if (result.response.method) {
                            const methodSection = document.createElement('div');
                            methodSection.className = 'response-section';
                            
                            const methodLabel = document.createElement('strong');
                            methodLabel.textContent = 'Methodology: ';
                            
                            const methodText = document.createElement('span');
                            methodText.textContent = result.response.method;
                            
                            methodSection.appendChild(methodLabel);
                            methodSection.appendChild(methodText);
                            messageContainer.appendChild(methodSection);
                        }
                        
                        // Confidence section
                        if (result.response.confidence !== undefined) {
                            const confidenceSection = document.createElement('div');
                            confidenceSection.className = 'response-section';
                            
                            const confidenceLabel = document.createElement('strong');
                            confidenceLabel.textContent = 'Confidence: ';
                            
                            const confidenceText = document.createElement('span');
                            confidenceText.textContent = result.response.confidence;
                            
                            confidenceSection.appendChild(confidenceLabel);
                            confidenceSection.appendChild(confidenceText);
                            messageContainer.appendChild(confidenceSection);
                        }
                        
                        // Category section
                        if (result.response.category && result.response.category !== 'Unknown') {
                            const categorySection = document.createElement('div');
                            categorySection.className = 'response-section';
                            
                            const categoryLabel = document.createElement('strong');
                            categoryLabel.textContent = 'Category: ';
                            
                            const categoryText = document.createElement('span');
                            categoryText.textContent = result.response.category;
                            
                            categorySection.appendChild(categoryLabel);
                            categorySection.appendChild(categoryText);
                            messageContainer.appendChild(categorySection);
                        }
                        
                        // Add Track Impact button if applicable
                        if (result.response.category && result.response.answer) {
                            // Use same carbon value extraction as above
                            const carbonPatterns = [
                                /(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\s*(?:of)?\s*(?:CO2e?|carbon dioxide equivalent)/i,
                                /(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of)?\s*(?:CO2e?|carbon dioxide equivalent)/i,
                                /(\d+(?:\.\d+)?)\s*(?:tons?|tonnes?)\s*(?:of)?\s*(?:CO2e?|carbon dioxide equivalent)/i,
                                /(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)/i,
                                /(\d+(?:\.\d+)?)\s*(?:g|grams?)/i,
                                /(\d+(?:\.\d+)?)\s*(?:CO2e?)/i,
                                /carbon footprint of (\d+(?:\.\d+)?)/i,
                                /(\d+(?:\.\d+)?)/
                            ];
                            
                            let carbonValue = null;
                            let unit = 'kg';
                            
                            for (const pattern of carbonPatterns) {
                                const match = result.response.answer.match(pattern);
                                if (match) {
                                    carbonValue = parseFloat(match[1]);
                                    
                                    if (pattern.source.includes('grams?') && !pattern.source.includes('kg')) {
                                        carbonValue = carbonValue / 1000;
                                        unit = 'kg (converted from g)';
                                    } else if (pattern.source.includes('tons?|tonnes?')) {
                                        carbonValue = carbonValue * 1000;
                                        unit = 'kg (converted from tons)';
                                    }
                                    
                                    console.log(`[Voice] Extracted carbon value: ${carbonValue} ${unit} using pattern: ${pattern}`);
                                    break;
                                }
                            }
                            
                            if (carbonValue !== null) {
                                const trackButton = document.createElement('button');
                                trackButton.className = 'track-query-btn';
                                trackButton.innerHTML = '<i class="fas fa-chart-line"></i> Track Impact';
                                
                                trackButton.addEventListener('click', function(e) {
                                    trackQuery(result.response.answer, result.response.category, carbonValue, e);
                                });
                                
                                messageContainer.appendChild(trackButton);
                            }
                        }
                    } else {
                        // Generic object, convert to string
                        messageContainer.textContent = JSON.stringify(result.response);
                    }
                } else {
                    // Fallback for any other type
                    messageContainer.textContent = String(result.response);
                }
                
                // Add to conversation and scroll
                conversationContainer.appendChild(messageContainer);
            conversationContainer.scrollTop = conversationContainer.scrollHeight;
            }
            
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
            
            // Also clean up thoughts
            console.log('[Voice] Cleaning up thoughts');
            fetch('/api/cleanup-thoughts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .catch(error => {
                console.error('[Voice] Error cleaning up thoughts:', error);
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
