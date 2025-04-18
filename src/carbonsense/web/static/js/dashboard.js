document.addEventListener('DOMContentLoaded', function() {
    // Initialize circular progress bars
    initCircularProgress();
    
    // Initialize weekly trends chart
    initWeeklyChart();
    
    // Setup chat functionality
    setupChatFunctionality();
});

// Initialize the circular progress indicators
function initCircularProgress() {
    const progressElements = document.querySelectorAll('.circular-progress');
    
    progressElements.forEach(element => {
        const dataValue = element.getAttribute('data-value');
        element.style.setProperty('--progress', `${dataValue}%`);
    });
}

// Initialize weekly trends chart using Chart.js
function initWeeklyChart() {
    const ctx = document.getElementById('weeklyChart').getContext('2d');
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Carbon Footprint (kg CO2)',
                data: weeklyData, // This comes from the template as passed from the backend
                fill: false,
                borderColor: '#2563eb',
                tension: 0.4,
                borderWidth: 3,
                pointBackgroundColor: '#ffffff',
                pointBorderColor: '#2563eb',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 14
                    },
                    padding: 10,
                    cornerRadius: 6,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `${context.raw} kg CO2e`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    display: false,
                    beginAtZero: true
                }
            },
            elements: {
                line: {
                    tension: 0.4
                }
            }
        }
    });
}

// Setup chat functionality
function setupChatFunctionality() {
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const conversationContainer = document.getElementById('conversation-container');
    
    // Handle sending messages
    function sendMessage() {
        const query = queryInput.value.trim();
        if (!query) return;
        
        console.log("==========================================");
        console.log("SENDING QUERY:", query);
        console.log("==========================================");
        
        // Add user message to the chat
        addMessageToChat(query, 'user');
        
        // Show loading indicator
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('message', 'bot', 'loading');
        loadingElement.textContent = 'Processing your request...';
        conversationContainer.appendChild(loadingElement);
        scrollToBottom();
        
        // Clear input field
        queryInput.value = '';
        
        // Disable input while processing
        queryInput.disabled = true;
        sendButton.disabled = true;
        
        // Setup EventSource for SSE (Server-Sent Events) to get real-time thoughts
        console.log("Starting thought streaming for query:", query);
        setupThoughtStream(query);
        console.log("Thought streaming initiated");
        
        // Call the API with the query
        console.log("Sending query to API:", query);
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            if (document.contains(loadingElement)) {
                conversationContainer.removeChild(loadingElement);
            }
            
            // Add bot response to the chat
            const answer = data.response || data.answer || "Sorry, I couldn't process that query.";
            addMessageToChat(answer, 'bot');
            
            // If sources are provided, display them
            if (data.sources && data.sources.length > 0) {
                const sourcesElement = document.createElement('div');
                sourcesElement.classList.add('message', 'bot', 'sources');
                
                const sourcesTitle = document.createElement('div');
                sourcesTitle.classList.add('sources-title');
                sourcesTitle.textContent = 'Sources:';
                sourcesElement.appendChild(sourcesTitle);
                
                const sourcesList = document.createElement('ul');
                data.sources.forEach(source => {
                    const sourceItem = document.createElement('li');
                    sourceItem.textContent = source;
                    sourcesList.appendChild(sourceItem);
                });
                sourcesElement.appendChild(sourcesList);
                
                conversationContainer.appendChild(sourcesElement);
            }
            
            // If the query looks like an activity, update metrics
            if (isActivity(query)) {
                updateActivityMetrics(query);
            }
            
            // Re-enable input
            queryInput.disabled = false;
            sendButton.disabled = false;
            queryInput.focus();
        })
        .catch(error => {
            // Remove loading indicator
            if (document.contains(loadingElement)) {
                conversationContainer.removeChild(loadingElement);
            }
            
            console.error('Error:', error);
            addMessageToChat('Sorry, there was an error processing your request. Please try again.', 'bot');
            
            // Re-enable input
            queryInput.disabled = false;
            sendButton.disabled = false;
        });
    }
    
    // Setup Server-Sent Events for streaming thoughts
    function setupThoughtStream(query) {
        try {
            // If there's already an active EventSource, close it
            if (window.activeEventSource) {
                window.activeEventSource.close();
                window.activeEventSource = null;
            }
            
            // Create the EventSource with the proper URL
            const eventSourceUrl = `/api/stream-thoughts?query=${encodeURIComponent(query)}`;
            
            const eventSource = new EventSource(eventSourceUrl);
            window.activeEventSource = eventSource;
            
            // Set max event listeners to avoid memory issues
            eventSource._maxListeners = 10;
            
            // Counter for connection attempts
            let connectionAttempts = 0;
            const MAX_RECONNECT_ATTEMPTS = 3;
            
            // Handle connection open
            eventSource.onopen = function(event) {
                connectionAttempts = 0; // Reset on successful connection
            };
            
            // Handle basic message events
            eventSource.onmessage = function(event) {
                if (!event.data || event.data.trim() === '') {
                    return;
                }
                
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'thought') {
                        addThoughtToChat(data.content);
                    } else if (data.type === 'action') {
                        addActionToChat(data.content);
                    } else if (data.type === 'complete') {
                        cleanupEventSource(eventSource);
                    } else if (data.type === 'error') {
                        console.error("Error from server:", data.content);
                        addThoughtToChat("Error: " + data.content);
                        // Don't close on error messages from server; they're just notifications
                    }
                } catch (error) {
                    console.error("Error processing event data:", error.message);
                    // Only add to chat for parsing errors
                    addThoughtToChat("Error processing server message. Please try again.");
                    cleanupEventSource(eventSource);
                }
            };
            
            // Handle errors
            eventSource.onerror = function(error) {
                if (eventSource.readyState === EventSource.CONNECTING) {
                    connectionAttempts++;
                    if (connectionAttempts <= MAX_RECONNECT_ATTEMPTS) {
                        addThoughtToChat("Connection interrupted. Reconnecting... (Attempt " + connectionAttempts + ")");
                    } else {
                        addThoughtToChat("Connection failed after multiple attempts. Please try again later.");
                        cleanupEventSource(eventSource);
                    }
                } else if (eventSource.readyState === EventSource.CLOSED) {
                    addThoughtToChat("Connection closed.");
                    cleanupEventSource(eventSource);
                }
            };
            
            // Close the connection after a timeout to prevent it hanging
            setTimeout(() => {
                if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
                    addThoughtToChat("Thought streaming timed out.");
                    cleanupEventSource(eventSource);
                }
            }, 30000); // 30 seconds timeout
            
            // Helper function to clean up EventSource
            function cleanupEventSource(es) {
                if (es && es.readyState !== EventSource.CLOSED) {
                    es.close();
                }
                if (window.activeEventSource === es) {
                    window.activeEventSource = null;
                }
            }
            
            return true;
        } catch (error) {
            console.error("Error setting up thought stream:", error.message);
            addThoughtToChat("Failed to connect to thought stream");
            
            if (window.activeEventSource) {
                window.activeEventSource.close();
                window.activeEventSource = null;
            }
            return false;
        }
    }
    
    // Add a message to the chat container
    function addMessageToChat(message, type) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', type);
        messageElement.textContent = message;
        
        conversationContainer.appendChild(messageElement);
        scrollToBottom();
    }
    
    // Add an agent thought to the chat
    function addThoughtToChat(thought) {
        // Don't add empty thoughts
        if (!thought || thought.trim() === '') {
            console.warn("Empty thought received, skipping");
            return;
        }
        
        console.log("ADDING THOUGHT:", thought);
        
        try {
            // Get conversation container (direct DOM access)
            const conversationContainer = document.getElementById('conversation-container');
            if (!conversationContainer) {
                console.error("ERROR: conversation-container not found in DOM!");
                return;
            }
            
            console.log("Creating thought element");
            // Create thought element
            const thoughtElement = document.createElement('div');
            thoughtElement.classList.add('message', 'thought');
            
            // Create thought header
            const thoughtHeader = document.createElement('div');
            thoughtHeader.classList.add('thought-header');
            thoughtHeader.innerHTML = '<i class="fas fa-brain"></i> Agent Thinking:';
            thoughtElement.appendChild(thoughtHeader);
            
            // Create thought content
            const thoughtContent = document.createElement('div');
            thoughtContent.classList.add('thought-content');
            thoughtContent.textContent = thought;
            thoughtElement.appendChild(thoughtContent);
            
            // Add to DOM
            console.log("Appending to conversation container");
            conversationContainer.appendChild(thoughtElement);
            
            // Force DOM update with explicit style change
            thoughtElement.style.opacity = '0';
            setTimeout(() => {
                thoughtElement.style.opacity = '1';
                // Scroll to bottom after opacity transition
                scrollToBottom();
            }, 10);
            
            console.log("Thought added to DOM:", thoughtElement);
            
            // Verify the thought was added
            setTimeout(() => {
                const allThoughts = document.querySelectorAll('.thought');
                console.log(`Total thoughts in DOM: ${allThoughts.length}`);
            }, 100);
        } catch (error) {
            console.error("ERROR adding thought to chat:", error);
        }
    }
    
    // Add an agent action to the chat
    function addActionToChat(action) {
        // Don't add empty actions
        if (!action || action.trim() === '') return;
        
        const actionElement = document.createElement('div');
        actionElement.classList.add('message', 'action');
        actionElement.innerHTML = `<strong>Action:</strong> ${action}`;
        
        conversationContainer.appendChild(actionElement);
        scrollToBottom();
    }
    
    // Helper to scroll to bottom of chat
    function scrollToBottom() {
        if (conversationContainer) {
            conversationContainer.scrollTop = conversationContainer.scrollHeight;
        }
    }
    
    // Check if a query looks like an activity
    function isActivity(query) {
        // Simple check - can be expanded based on your activity patterns
        const activityPhrases = ['i drove', 'i ate', 'i used', 'i consumed', 'i bought', 'i took', 'i had'];
        return activityPhrases.some(phrase => query.toLowerCase().includes(phrase));
    }
    
    // Update metrics when user enters an activity
    function updateActivityMetrics(activity) {
        fetch('/api/activity', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ activity: activity })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Activity recorded:', data);
            
            // If we got analysis back, show it in the chat
            if (data.analysis && data.analysis !== "Activity recorded, but analysis is currently unavailable.") {
                // Add a small delay so it appears after the main response
                setTimeout(() => {
                    addMessageToChat(`Carbon Impact Analysis: ${data.analysis}`, 'bot');
                }, 500);
                
                // In a real app, you would update the dashboard metrics here
                // This would require additional API endpoints to get updated data
                // For now, we'll just simulate an update after 1 second
                setTimeout(updateDashboardMetrics, 1000);
            }
        })
        .catch(error => {
            console.error('Error recording activity:', error);
        });
    }
    
    // Simulate updating the dashboard metrics
    function updateDashboardMetrics() {
        // This is just a simulation - in a real app, you would fetch
        // the latest metrics from the server and update the display
        
        // Get random increase for total carbon (between 0.1 and 1.5)
        const increase = (Math.random() * 1.4 + 0.1).toFixed(1);
        
        // Get the current total
        const totalElement = document.querySelector('.metric-card:first-child .value');
        const currentTotal = parseFloat(totalElement.textContent);
        
        // Update the total
        const newTotal = (currentTotal + parseFloat(increase)).toFixed(1);
        totalElement.textContent = newTotal;
        
        // Update the progress circle
        const progressCircle = document.querySelector('.metric-card:first-child .circular-progress');
        progressCircle.setAttribute('data-value', (newTotal / 10 * 100));
        progressCircle.style.setProperty('--progress', `${newTotal / 10 * 100}%`);
        
        // Flash the card to indicate an update
        const card = document.querySelector('.metric-card:first-child');
        card.classList.add('updated');
        setTimeout(() => {
            card.classList.remove('updated');
        }, 1000);
    }
    
    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    
    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Add a welcome message
    setTimeout(() => {
        addMessageToChat('Hello! Ask me anything about carbon footprints or tell me about your daily activities to track your carbon impact.', 'bot');
    }, 500);
} 