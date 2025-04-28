// CarbonSense Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('[Dashboard] Initializing dashboard components');
    
    // Check if the API is available
    if (window.CarbonSenseAPI) {
        console.log('[Dashboard] CarbonSenseAPI is available with methods:', 
            Object.keys(window.CarbonSenseAPI).join(', '));
    } else {
        console.error('[Dashboard] CarbonSenseAPI is NOT available');
    }
    
    // Initialize dashboard components that exist in the DOM
    initializeCategoryPieChart();
    setupPieChartModal();
    setupChatInteractions();
    setupInfoButtons();
    setupExamplePills();
    setupPeriodSelector();
    
    // Add CSS styles for thought streaming
    addThoughtStreamingStyles();
});

// Add styles for thought streaming
function addThoughtStreamingStyles() {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .thinking-container {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            background: rgba(240, 240, 255, 0.7);
            transition: all 0.3s ease;
            overflow: hidden;
        }
        
        .thinking-container.collapsed {
            max-height: 40px;
            overflow: hidden;
        }
        
        .thinking-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-weight: bold;
            color: #424242;
        }
        
        .thoughts-area {
            max-height: 200px;
            overflow-y: auto;
            padding-right: 5px;
        }
        
        .thought {
            padding: 6px 10px;
            margin: 5px 0;
            border-radius: 6px;
            display: flex;
            align-items: center;
            animation: fadeIn 0.5s ease;
        }
        
        .thought.thought {
            background: rgba(173, 216, 230, 0.3);
        }
        
        .thought.action {
            background: rgba(144, 238, 144, 0.3);
        }
        
        .thought.error {
            background: rgba(255, 182, 193, 0.3);
        }
        
        .thought.complete {
            background: rgba(152, 251, 152, 0.4);
            font-weight: bold;
        }
        
        .thought-icon {
            margin-right: 8px;
            color: #3f51b5;
        }
        
        .toggle-thinking-btn {
            border: none;
            background: none;
            color: #3f51b5;
            cursor: pointer;
            font-size: 0.8em;
            padding: 3px 8px;
            border-radius: 4px;
            transition: background 0.3s ease;
        }
        
        .toggle-thinking-btn:hover {
            background: rgba(63, 81, 181, 0.1);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }
    `;
    document.head.appendChild(styleElement);
}

// Set up info buttons to show tooltips
function setupInfoButtons() {
    const infoButtons = document.querySelectorAll('.chart-tip');
    if (infoButtons.length === 0) return;
    
    infoButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tipText = this.getAttribute('data-tip');
            
            // Create or update tooltip
            let tooltip = document.querySelector('.info-tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.className = 'info-tooltip';
                document.body.appendChild(tooltip);
            }
            
            tooltip.textContent = tipText;
            tooltip.style.display = 'block';
            tooltip.style.left = `${this.getBoundingClientRect().left - tooltip.offsetWidth/2 + this.offsetWidth/2}px`;
            tooltip.style.top = `${this.getBoundingClientRect().bottom + 10}px`;
            
            // Hide tooltip after 3 seconds
            setTimeout(() => {
                tooltip.style.display = 'none';
            }, 3000);
        });
    });
}

// Set up chat interactions
function setupChatInteractions() {
    const sendButton = document.getElementById('send-button');
    const queryInput = document.getElementById('query-input');
    const conversationContainer = document.getElementById('conversation-container');
    
    if (!sendButton || !queryInput || !conversationContainer) return;
    
    // Initialize with a welcome message if conversation is empty
    if (conversationContainer.children.length === 0) {
        addBotMessage("Hello! I'm your CarbonSense AI assistant. How can I help you reduce your carbon footprint today?");
    }
    
    sendButton.addEventListener('click', function() {
        sendMessage();
    });
    
    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    function sendMessage() {
        const message = queryInput.value.trim();
        if (message === '') return;
        
        console.log('[Dashboard] Sending message:', message);
        
        // Add user message to conversation
        addUserMessage(message);
        
        // Clear input
        queryInput.value = '';
        
        // Create thinking container for thoughts
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
        
        // Initialize polling variables
        let thoughtPollingId = null;
        let lastSeenThoughtIndex = -1;
        let pollCount = 0;
        let isPolling = false;
        let currentThoughtType = 'thought';
        
        // Start independent thought polling immediately
        startThoughtPolling();
        
        // Make actual query in parallel - use direct fetch if API not available
        if (window.CarbonSenseAPI && window.CarbonSenseAPI.queryCarbonFootprint) {
            window.CarbonSenseAPI.queryCarbonFootprint(message)
                .then(handleQueryResult)
                .catch(handleQueryError);
        } else {
            // Fallback to direct fetch
            console.log('[Dashboard] Using direct fetch for query');
            fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: message })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error: ${response.status}`);
                }
                return response.json();
            })
            .then(handleQueryResult)
            .catch(handleQueryError);
        }
        
        // Function to start thought polling - independent of backend response
        function startThoughtPolling() {
            console.log('[Dashboard] Starting independent thought polling');
            thoughtPollingId = {};  // Use an object as a non-null identifier
            pollForThoughts();
        }
        
        // Function to poll for thoughts independently
        async function pollForThoughts() {
            // Prevent multiple concurrent polling
            if (isPolling) return;
            
            isPolling = true;
            pollCount++;
            
            try {
                console.log('[Dashboard] Polling for thoughts directly (poll #' + pollCount + ')');
                
                // Make direct fetch request to backend
                const response = await fetch('/api/check-thoughts');
                if (!response.ok) {
                    throw new Error(`HTTP error: ${response.status}`);
                }
                
                const thoughtsData = await response.json();
                console.log('[Dashboard] Received thoughts data:', thoughtsData);
                    
                // Process the thoughts if we have any
                if (thoughtsData && thoughtsData.thoughts && thoughtsData.thoughts.length > 0) {
                    // Process only new thoughts
                    const newThoughts = thoughtsData.thoughts.slice(lastSeenThoughtIndex + 1);
                    
                    if (newThoughts.length > 0) {
                        console.log('[Dashboard] Found new thoughts:', newThoughts.length);
                        
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
                    if (thoughtsData.status === 'complete') {
                        console.log('[Dashboard] Thought process complete');
                        
                        // Add final thought
                        updateThoughtElement(thoughtElement, {
                            content: 'Analysis complete!',
                            type: 'complete'
                        });
                        
                        // Scroll thoughts area down
                        thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                        
                        // Stop polling when complete
                        stopPolling();
                        return; // Exit the function early
                    } else if (thoughtsData.status === 'error') {
                        console.log('[Dashboard] Thought process error');
                        
                        // Update with error thought
                        updateThoughtElement(thoughtElement, {
                            content: 'Error in thought processing',
                            type: 'error'
                        });
                        
                        // Scroll thoughts area down
                        thoughtsArea.scrollTop = thoughtsArea.scrollHeight;
                        
                        // Stop polling when there's an error
                        stopPolling();
                        return; // Exit the function early
                    }
                } else {
                    console.log('[Dashboard] No thoughts available yet');
                }
            } catch (error) {
                console.error('[Dashboard] Error fetching thoughts:', error);
            } finally {
                isPolling = false;
                
                // Continue polling with a delay regardless of backend response
                if (thoughtPollingId !== null) {
                    // Use exponential backoff for polling frequency
                    const delayMs = Math.min(1000 * Math.pow(1.2, Math.min(pollCount, 10)), 5000);
                    console.log(`[Dashboard] Next poll in ${delayMs/1000} seconds`);
                    
                    setTimeout(() => {
                        if (thoughtPollingId !== null) {  // Check again in case stopped during timeout
                            pollForThoughts();
                        }
                    }, delayMs);
                }
            }
        }
        
        // Function to update the thought element
        function updateThoughtElement(element, thought) {
            // Get thought type or default to 'thought'
            const thoughtType = thought.type || 'thought';
            
            // Only change the class if the type has changed
            if (currentThoughtType !== thoughtType) {
                // Update element class
                element.className = `thought ${thoughtType}`;
                currentThoughtType = thoughtType;
                
                // Update icon based on type
                let icon = '';
                switch(thoughtType) {
                    case 'thought':
                        icon = '<i class="fas fa-brain"></i>';
                        break;
                    case 'action':
                        icon = '<i class="fas fa-cog fa-spin"></i>';
                        break;
                    case 'error':
                        icon = '<i class="fas fa-exclamation-triangle"></i>';
                        break;
                    case 'complete':
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
            
            console.log(`[Dashboard] Updated thought: ${content} (${thoughtType})`);
        }
        
        function stopPolling() {
            console.log('[Dashboard] Stopping thought polling');
            
            if (thoughtPollingId !== null) {
                thoughtPollingId = null;
                // No longer clean up thoughts - we want to keep them
                // fetch('/api/cleanup-thoughts', {
                //     method: 'POST'
                // }).catch(error => {
                //     console.error('[Dashboard] Error cleaning up thoughts:', error);
                // });
            }
        }
        
        // Handler for successful query results - independent of thought polling
        function handleQueryResult(result) {
            console.log('[Dashboard] Query result:', result);
            
            // Handle the final result
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
            
            // Add bot response
            addBotMessage(result);
            
            // Stop polling since query is complete
            stopPolling();
        }
        
        // Handler for query errors - independent of thought polling
        function handleQueryError(error) {
            console.error('[Dashboard] Error querying API:', error);
            
            // Add error message to the conversation
            addBotMessage("I'm unable to process your request at this time. Please try again later.");
            
            // Stop polling since there was an error
            stopPolling();
        }
    }
    
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user';
        messageElement.textContent = message;
        conversationContainer.appendChild(messageElement);
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
    }
}

// Helper function to add a thought to the thoughts area
function addThought(container, thought, type) {
    console.log(`[Dashboard] Adding thought to UI: ${thought.content} (${type})`);
    const thoughtElement = document.createElement('div');
    thoughtElement.className = `thought ${type || 'thought'}`;
    
    // Add icon based on type
    let icon = '';
    switch(type) {
        case 'thought':
            icon = '<i class="fas fa-brain"></i>';
            break;
        case 'action':
            icon = '<i class="fas fa-cog fa-spin"></i>';
            break;
        case 'error':
            icon = '<i class="fas fa-exclamation-triangle"></i>';
            break;
        case 'complete':
            icon = '<i class="fas fa-check-circle"></i>';
            break;
        default:
            icon = '<i class="fas fa-comment"></i>';
    }
    
    // Ensure thought.content exists before using it
    const content = thought.content || 'No content available';
    
    thoughtElement.innerHTML = `
        <span class="thought-icon">${icon}</span>
        <span class="thought-content">${content}</span>
    `;
    
    container.appendChild(thoughtElement);
    
    // Ensure the container is scrolled to the bottom
    container.scrollTop = container.scrollHeight;
}

// Make addUserMessage available globally for the voice handler
window.addUserMessage = function(message) {
    const conversationContainer = document.getElementById('conversation-container');
    if (!conversationContainer) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = 'message user';
    messageElement.textContent = message;
    conversationContainer.appendChild(messageElement);
    conversationContainer.scrollTop = conversationContainer.scrollHeight;
};

// Set up example pills for quick chat queries
function setupExamplePills() {
    const examplePills = document.querySelectorAll('.example-pill');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    
    if (examplePills.length === 0 || !queryInput || !sendButton) return;
    
    examplePills.forEach(pill => {
        pill.addEventListener('click', function() {
            const exampleText = this.textContent;
            queryInput.value = exampleText;
            queryInput.focus();
            
            // Trigger send after a short delay
                setTimeout(() => {
                sendButton.click();
            }, 300);
        });
    });
}

// Setup for time period selector
function setupPeriodSelector() {
    const periodSelector = document.getElementById('time-period-selector');
    const totalBox = document.getElementById('carbon-total-box');
    
    if (!periodSelector || !totalBox) return;
    
    // Sample data for different time periods
    const periodData = {
        daily: {
            title: "Today's Total",
            value: "8.2",
            target: "12.15",
            categories: {
                food: "2.3",
                energy: "1.7",
                mobility: "1.5",
                purchases: "2.5",
                misc: "2.0"
            }
        },
        weekly: {
            title: "Weekly Total",
            value: "43.7",
            target: "44.1",
            categories: {
                food: "2.1",
                energy: "1.8",
                mobility: "1.6",
                purchases: "2.2",
                misc: "0.7"
            }
        },
        monthly: {
            title: "Monthly Total",
            value: "172.4",
            target: "180",
            categories: {
                food: "8.4",
                energy: "7.2",
                mobility: "6.8",
                purchases: "8.9",
                misc: "3.1"
            }
        }
    };
    
    periodSelector.addEventListener('change', function() {
        const period = this.value;
        const data = periodData[period];
        
        if (!data) return;
        
        // Update total box
        totalBox.querySelector('.stat-title').textContent = data.title;
        totalBox.querySelector('.stat-value').textContent = `${data.value} kg CO2`;
        totalBox.querySelector('.stat-target').textContent = `Target: ${data.target} kg`;
        
        // Calculate percentage for change indicator
        const currentValue = parseFloat(data.value);
        const targetValue = parseFloat(data.target);
        const percentChange = ((Math.abs(currentValue - targetValue) / targetValue) * 100).toFixed(1);
        
        const changeIndicator = totalBox.querySelector('.stat-change');
        if (currentValue <= targetValue) {
            changeIndicator.textContent = `-${percentChange}%`;
            changeIndicator.classList.remove('negative');
            changeIndicator.classList.add('positive');
        } else {
            changeIndicator.textContent = `+${percentChange}%`;
            changeIndicator.classList.remove('positive');
            changeIndicator.classList.add('negative');
        }
        
        // Update category legend
        updateCategoryLegend(data.categories);
        
        // Update pie chart
        updateCategoryPieChart(data.categories);
    });
}

// Update category legend with new data
function updateCategoryLegend(categories) {
    const legend = document.getElementById('category-legend');
    if (!legend) return;
    
    const legendItems = legend.querySelectorAll('.legend-item');
    const categoryKeys = Object.keys(categories);
    
    legendItems.forEach((item, index) => {
        if (index < categoryKeys.length) {
            const category = categoryKeys[index];
            const value = categories[category];
            item.querySelector('.legend-value').textContent = `${value} kg`;
            }
        });
    }
    
// Update category pie chart with new data
function updateCategoryPieChart(categories) {
    const chart = Chart.getChart('category-pie-chart');
    if (!chart) return;
    
    // Extract values in the correct order
    const values = [
        parseFloat(categories.food),
        parseFloat(categories.energy),
        parseFloat(categories.mobility),
        parseFloat(categories.purchases),
        parseFloat(categories.misc)
    ];
    
    chart.data.datasets[0].data = values;
    chart.update();
}

// Initialize category pie chart
function initializeCategoryPieChart() {
    const categoryChart = document.getElementById('category-pie-chart');
    if (!categoryChart) return;
    
    // Get category values from the DOM with null checks
    const getValueFromLegend = (index) => {
        const element = document.querySelector(`.legend-item:nth-child(${index}) .legend-value`);
        return element ? parseFloat(element.textContent) || 0 : 0;
    };

    const foodValue = getValueFromLegend(1);
    const energyValue = getValueFromLegend(2);
    const mobilityValue = getValueFromLegend(3);
    const purchasesValue = getValueFromLegend(4);
    const miscValue = getValueFromLegend(5);
        
    // Create the pie chart
    const pieChart = new Chart(categoryChart, {
            type: 'pie',
            data: {
                labels: ['Food & Diet', 'Energy Use', 'Mobility', 'Purchases', 'Miscellaneous'],
                datasets: [{
                    data: [foodValue, energyValue, mobilityValue, purchasesValue, miscValue],
                    backgroundColor: [
                        '#4CAF50', // Green for Food
                        '#2196F3', // Blue for Energy
                        '#9C27B0', // Purple for Mobility
                        '#FFC107', // Yellow for Purchases
                        '#9E9E9E'  // Gray for Miscellaneous
                    ],
                    borderWidth: 1,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false // Hide default legend since we're using custom legend
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                // Don't include zero values in percentage calculation
                                const nonZeroData = context.dataset.data.filter(val => val > 0);
                                const total = nonZeroData.reduce((a, b) => a + b, 0);
                                const percentage = Math.round((value / total) * 100);
                                return `${label}: ${value} kg (${percentage}%)`;
                            }
                        }
                    }
                },
                backgroundColor: 'white'
            }
    });
    
    // Make chart instance available globally
    window.categoryPieChart = pieChart;
    return pieChart;
}

// Setup pie chart modal functionality
function setupPieChartModal() {
    // Setup for category pie chart modal
    const pieChartContainer = document.querySelector('.pie-chart-container');
    const pieChartModal = document.getElementById('pie-chart-modal');
    const closeModalBtn = pieChartModal?.querySelector('.close-modal');
    
    if (pieChartContainer && pieChartModal) {
        pieChartContainer.addEventListener('click', function() {
            pieChartModal.style.display = 'flex'; // Using flex to center the modal content
            initializeZoomedPieChart();
            
            // Prevent body scrolling when modal is open
            document.body.style.overflow = 'hidden';
        });
    }
    
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', function() {
            pieChartModal.style.display = 'none';
            // Restore body scrolling
            document.body.style.overflow = 'auto';
        });
    }
    
    // Close modal when clicking outside the content
    window.addEventListener('click', function(event) {
        if (event.target === pieChartModal) {
            pieChartModal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    });
}

// Initialize the zoomed pie chart in the modal
function initializeZoomedPieChart() {
    const zoomedChart = document.getElementById('zoomed-category-pie-chart');
    if (!zoomedChart) return;
    
    // Get category values from the DOM with null checks
    const getValueFromLegend = (index) => {
        const element = document.querySelector(`.legend-item:nth-child(${index}) .legend-value`);
        return element ? parseFloat(element.textContent) || 0 : 0;
    };

    const foodValue = getValueFromLegend(1);
    const energyValue = getValueFromLegend(2);
    const mobilityValue = getValueFromLegend(3);
    const purchasesValue = getValueFromLegend(4);
    const miscValue = getValueFromLegend(5);
    
    // Create the zoomed pie chart
    new Chart(zoomedChart, {
        type: 'pie',
        data: {
            labels: ['Food & Diet', 'Energy Use', 'Mobility', 'Purchases', 'Miscellaneous'],
            datasets: [{
                data: [foodValue, energyValue, mobilityValue, purchasesValue, miscValue],
                backgroundColor: [
                    '#4CAF50', // Green for Food
                    '#2196F3', // Blue for Energy
                    '#9C27B0', // Purple for Mobility
                    '#FFC107', // Yellow for Purchases
                    '#9E9E9E'  // Gray for Miscellaneous
                ],
                borderWidth: 1,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        font: {
                            size: 14
                        },
                        padding: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            // Don't include zero values in percentage calculation
                            const nonZeroData = context.dataset.data.filter(val => val > 0);
                            const total = nonZeroData.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} kg (${percentage}%)`;
                        }
                    }
                }
            },
            // Set the background to white
            backgroundColor: 'white'
        }
    });
}

// Helper function for parsing nested JSON in API responses
function tryParseNestedJSON(jsonString) {
    if (typeof jsonString !== 'string') return null;
    
    // Skip if not likely to be JSON
    if (!jsonString.trim().startsWith('{') && !jsonString.trim().startsWith("{'")) return null;
    
    try {
        // Try standard JSON format first
        if (jsonString.trim().startsWith('{') && jsonString.trim().endsWith('}')) {
            return JSON.parse(jsonString);
        }
        
        // Try parsing Python-style dict with single quotes
        if (jsonString.trim().startsWith("{'") && jsonString.trim().endsWith("'}")) {
            const jsonStr = jsonString
                .replace(/'/g, '"')  // Replace single quotes with double quotes
                .replace(/None/g, 'null')  // Replace None with null
                .replace(/True/g, 'true')  // Replace True with true
                .replace(/False/g, 'false'); // Replace False with false
            
            return JSON.parse(jsonStr);
        }
    } catch (e) {
        console.error("Failed to parse nested JSON:", e);
    }
    
    return null;
}

// Helper function to check if an object looks like a valid carbon response
function isValidCarbonResponse(obj) {
    return obj && 
           typeof obj === 'object' && 
           typeof obj.answer !== 'undefined' && 
           typeof obj.method !== 'undefined' && 
           typeof obj.confidence !== 'undefined' && 
           typeof obj.category !== 'undefined';
}

// Add bot message to the conversation
function addBotMessage(result) {
    const conversationContainer = document.getElementById('conversation-container');
    
    // Create the main message container
    const messageContainer = document.createElement('div');
    messageContainer.className = 'message bot';
    
    // Check if result is a string or an object
    if (typeof result === 'string') {
        // Simple string response
        messageContainer.textContent = result;
    } else {
        // First, try to handle cases where answer itself contains nested JSON
        try {
            // Check for nested JSON in the answer field
            if (result.answer && typeof result.answer === 'string') {
                const parsedAnswer = tryParseNestedJSON(result.answer);
                
                // If we successfully parsed nested JSON and it looks like a valid response
                if (parsedAnswer && isValidCarbonResponse(parsedAnswer)) {
                    console.log("Using nested structured data from answer field");
                    result = parsedAnswer;
                }
            }
        } catch (e) {
            console.error("Error processing nested data:", e);
            // Continue with original result if parsing fails
        }
        
        // Structured response with the exact format requested
        
        // Create answer paragraph with label
        const answerSection = document.createElement('div');
        answerSection.className = 'response-section';
        
        const answerLabel = document.createElement('strong');
        answerLabel.textContent = 'Answer: ';
        
        const answerText = document.createElement('span');
        answerText.textContent = result.answer;
        
        answerSection.appendChild(answerLabel);
        answerSection.appendChild(answerText);
        messageContainer.appendChild(answerSection);
        
        // Method/Methodology section
        if (result.method) {
            const methodSection = document.createElement('div');
            methodSection.className = 'response-section';
            
            const methodLabel = document.createElement('strong');
            methodLabel.textContent = 'Methodology: ';
            
            const methodText = document.createElement('span');
            methodText.textContent = result.method;
            
            methodSection.appendChild(methodLabel);
            methodSection.appendChild(methodText);
            messageContainer.appendChild(methodSection);
        }
        
        // Confidence section
        if (result.confidence !== undefined) {
            const confidenceSection = document.createElement('div');
            confidenceSection.className = 'response-section';
            
            const confidenceLabel = document.createElement('strong');
            confidenceLabel.textContent = 'Confidence: ';
            
            const confidenceText = document.createElement('span');
            confidenceText.textContent = result.confidence;
            
            confidenceSection.appendChild(confidenceLabel);
            confidenceSection.appendChild(confidenceText);
            messageContainer.appendChild(confidenceSection);
        }
        
        // Category section
        if (result.category && result.category !== 'Unknown') {
            const categorySection = document.createElement('div');
            categorySection.className = 'response-section';
            
            const categoryLabel = document.createElement('strong');
            categoryLabel.textContent = 'Category: ';
            
            const categoryText = document.createElement('span');
            categoryText.textContent = result.category;
            
            categorySection.appendChild(categoryLabel);
            categorySection.appendChild(categoryText);
            messageContainer.appendChild(categorySection);
        }

        // Add Track Impact button if we have a valid category and numeric value in the answer
        if (result.category && result.answer) {
            // Try to extract numeric value from the answer using a more precise regex
            // Look for patterns like "X kg CO2e", "X grams of CO2", etc.
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
                const match = result.answer.match(pattern);
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
                    
                    console.log(`Extracted carbon value: ${carbonValue} ${unit} using pattern: ${pattern}`);
                    break;
                }
            }
            
            if (carbonValue !== null) {
                const trackButton = document.createElement('button');
                trackButton.className = 'track-query-btn';
                trackButton.innerHTML = '<i class="fas fa-chart-line"></i> Track Impact';
                
                // Add click handler with proper event passing
                trackButton.addEventListener('click', function(e) {
                    // Make sure the event is properly passed
                    console.log('Track button clicked, passing event:', e);
                    trackQuery(result.answer, result.category, carbonValue, e);
                });
                
                messageContainer.appendChild(trackButton);
            }
        }
    }
    
    // Add to conversation
    conversationContainer.appendChild(messageContainer);
    conversationContainer.scrollTop = conversationContainer.scrollHeight;
}

// Make sure trackQuery is globally available for both text and voice interactions
window.trackQuery = function(query, category, carbonValue, event) {
    // Prevent event propagation
    if (event) {
        event.stopPropagation();
    }
    
    console.log(`Tracking query: "${query}" in category: ${category} with value: ${carbonValue}`);
    
    // Make API call to track the query
    fetch('/api/track-query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: query,
            category: category,
            carbon_value: carbonValue
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Updated dashboard data:', data);
        
        // Update the dashboard with the new data
        updateDashboardData(data);
        
        // Show success notification
        showTrackingNotification(true, `Added ${carbonValue.toFixed(2)} kg CO₂ to your ${category} footprint.`);
    })
    .catch(error => {
        console.error('Error tracking query:', error);
        showTrackingNotification(false, 'Failed to track this item. Please try again.');
    });
};

// Helper function to show tracking notification
function showTrackingNotification(isSuccess, message) {
    // Create notification element if it doesn't exist
    let notification = document.querySelector('.tracking-notification');
    if (!notification) {
        notification = document.createElement('div');
        notification.className = 'tracking-notification';
        document.body.appendChild(notification);
    }
    
    // Set the style based on success/error
    notification.className = isSuccess ? 
        'tracking-notification success' : 
        'tracking-notification error';
    
    // Set the message
    notification.textContent = message;
    
    // Show the notification
    notification.style.display = 'block';
    notification.style.opacity = '1';
    
    // Hide after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            notification.style.display = 'none';
        }, 500);
    }, 3000);
}

// Update dashboard data with new values
function updateDashboardData(data) {
    // Update total carbon box
    const totalBox = document.getElementById('carbon-total-box');
    if (totalBox) {
        const totalValue = totalBox.querySelector('.stat-value');
        if (totalValue) {
            totalValue.textContent = `${data.total_carbon} kg CO2`;
        }
        
        // Update change indicator
        const target = 12.15; // Hardcoded target value from the template
        const percentChange = ((Math.abs(data.total_carbon - target) / target) * 100).toFixed(1);
        const changeIndicator = totalBox.querySelector('.stat-change');
        
        if (changeIndicator) {
            if (data.total_carbon <= target) {
                changeIndicator.textContent = `-${percentChange}%`;
                changeIndicator.classList.remove('negative');
                changeIndicator.classList.add('positive');
            } else {
                changeIndicator.textContent = `+${percentChange}%`;
                changeIndicator.classList.remove('positive');
                changeIndicator.classList.add('negative');
            }
        }
    }
    
    // Update category legend
    const legend = document.getElementById('category-legend');
    if (legend) {
        // Food & Diet
        const foodItem = legend.querySelector('.legend-item:nth-child(1) .legend-value');
        if (foodItem) foodItem.textContent = `${data.food_carbon} kg`;
        
        // Energy Use
        const energyItem = legend.querySelector('.legend-item:nth-child(2) .legend-value');
        if (energyItem) energyItem.textContent = `${data.household_carbon} kg`;
        
        // Mobility
        const mobilityItem = legend.querySelector('.legend-item:nth-child(3) .legend-value');
        if (mobilityItem) mobilityItem.textContent = `${data.transportation_carbon} kg`;
        
        // Purchases
        const purchasesItem = legend.querySelector('.legend-item:nth-child(4) .legend-value');
        if (purchasesItem) purchasesItem.textContent = `${data.goods_carbon} kg`;
        
        // Miscellaneous
        const miscItem = legend.querySelector('.legend-item:nth-child(5) .legend-value');
        if (miscItem) miscItem.textContent = `${data.misc_carbon} kg`;
    }
    
    // Update the pie chart
    const categories = {
        food: data.food_carbon,
        energy: data.household_carbon,
        mobility: data.transportation_carbon,
        purchases: data.goods_carbon,
        misc: data.misc_carbon
    };
    
    updateCategoryPieChart(categories);
}

// Make these functions available globally
window.addBotMessage = addBotMessage;
window.updateDashboardData = updateDashboardData;
window.showTrackingNotification = showTrackingNotification; 
window.addThought = addThought; 