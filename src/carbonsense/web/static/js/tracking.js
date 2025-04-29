// Global variable to store the pie chart instance
let categoryPieChart = null;

// Global variable to store the current period
let currentPeriod = "daily";

// Make trackQuery available globally
window.trackQuery = trackQuery;

// Function to update the pie chart with new data
function updatePieChart(updatedData) {
    console.log('Updating pie chart with data:', updatedData);
    
    // Return early if chart doesn't exist
    if (!categoryPieChart && !window.categoryPieChart) {
        console.log('Pie chart not initialized yet');
        return;
    }

    const chartData = {
        labels: ['Food & Diet', 'Energy Use', 'Mobility', 'Purchases', 'Miscellaneous'],
        datasets: [{
            data: [
                updatedData.food_carbon,
                updatedData.household_carbon,
                updatedData.transportation_carbon,
                updatedData.goods_carbon,
                updatedData.misc_carbon
            ],
            backgroundColor: [
                '#A7D7C5', // Green for Food
                '#FAE29C', // Blue for Energy
                '#64B5F6', // Purple for Mobility
                '#3B8686', // Yellow for Purchases
                '#F5B665'  // Gray for Miscellaneous
            ]
        }]
    };

    // Use either the local reference or the window reference
    const chart = categoryPieChart || window.categoryPieChart;
    if (chart) {
        console.log('Updating chart data:', chartData.datasets[0].data);
        chart.data = chartData;
        chart.update();
    }
}

// Function to update all dashboard values
function updateDashboardValues(updatedData) {
    console.log('Updating dashboard values with:', updatedData);
    
    // Update total carbon value
    const totalElement = document.querySelector('#carbon-total-box .stat-value');
    if (totalElement) {
        totalElement.textContent = `${updatedData.total_carbon.toFixed(2)} kg CO₂`;
        totalElement.classList.add('value-updated');
        setTimeout(() => totalElement.classList.remove('value-updated'), 300);
    }
    
    // Update time period display in the stat-title
    const periodName = updatedData.period ? updatedData.period.charAt(0).toUpperCase() + updatedData.period.slice(1) : 'Daily';
    const titleElement = document.querySelector('#carbon-total-box .stat-title');
    if (titleElement) {
        titleElement.textContent = `${periodName} Total`;
    }

    // Update category values in the legend
    const categories = {
        'food': updatedData.food_carbon,
        'energy': updatedData.household_carbon,
        'mobility': updatedData.transportation_carbon,
        'purchases': updatedData.goods_carbon,
        'miscellaneous': updatedData.misc_carbon
    };

    Object.entries(categories).forEach(([category, value]) => {
        const valueElement = document.querySelector(`.legend-item .legend-color.${category} + .legend-text .legend-value`);
        if (valueElement) {
            valueElement.textContent = `${value.toFixed(2)} kg`;
            valueElement.classList.add('value-updated');
            setTimeout(() => valueElement.classList.remove('value-updated'), 300);
        }
    });

    // Update target comparison
    // Adjust target values based on period
    let targetValue = 12.15; // Daily target
    if (updatedData.period === "weekly") {
        targetValue = 85.05; // Weekly target (daily * 7)
    } else if (updatedData.period === "monthly") {
        targetValue = 364.5; // Monthly target (daily * 30)
    }
    
    const totalCarbon = updatedData.total_carbon;
    const changeElement = document.querySelector('#carbon-total-box .stat-change');
    if (changeElement) {
        const percentChange = ((Math.abs(totalCarbon - targetValue) / targetValue) * 100).toFixed(1);
        if (totalCarbon > targetValue) {
            changeElement.innerHTML = `<i class="fas fa-arrow-circle-up"></i> ${percentChange}% over target`;
            changeElement.className = 'stat-change negative';
        } else {
            changeElement.innerHTML = `<i class="fas fa-arrow-circle-down"></i> ${percentChange}% under target`;
            changeElement.className = 'stat-change positive';
        }
    }
    
    // Update target display
    const targetElement = document.querySelector('#carbon-total-box .stat-target');
    if (targetElement) {
        targetElement.textContent = `Target: ${targetValue.toFixed(2)} kg`;
    }
    
    // If the pie chart modal is visible, update its data too
    const pieChartModal = document.getElementById('pie-chart-modal');
    if (pieChartModal && pieChartModal.style.display === 'flex') {
        // If modal is visible, update the zoomed pie chart
        console.log('[Tracking] Updating zoomed pie chart with period data');
        if (window.updateDetailedBreakdown) {
            window.updateDetailedBreakdown(
                updatedData.food_carbon,
                updatedData.household_carbon,
                updatedData.transportation_carbon,
                updatedData.goods_carbon,
                updatedData.misc_carbon
            );
        }
    }
}

// Enhanced trackQuery function
async function trackQuery(query, category, carbonValue, event) {
    try {
        console.log(`[Tracking] Tracking query in category: ${category}, with carbon value: ${carbonValue}, period: ${currentPeriod}`);
        console.log('[Tracking] This will update tracked_queries.json');
        
        // Validate the carbon value
        if (isNaN(carbonValue) || carbonValue <= 0) {
            console.error(`[Tracking] Invalid carbon value: ${carbonValue}`);
            showNotification('Invalid carbon value for tracking', 'error');
            return;
        }
        
        // Store button reference before fetch if event exists
        let trackButton = null;
        if (event && event.target) {
            trackButton = event.target.closest('.track-query-btn') || event.target;
            console.log('[Tracking] Found track button:', trackButton);
            
            // Disable button immediately to prevent multiple clicks
            trackButton.disabled = true;
            trackButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Tracking...';
        } else {
            console.log('[Tracking] No event or event.target provided');
        }
        
        const response = await fetch('/api/track-query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
            },
            body: JSON.stringify({
                query: query,
                category: category,
                carbon_value: carbonValue,
                period: currentPeriod
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to track query: ${response.status} ${response.statusText}`);
        }
        
        const updatedData = await response.json();
        console.log('[Tracking] Received updated data from tracked_queries.json:', updatedData);
        
        // Validate received data
        if (!updatedData || typeof updatedData !== 'object') {
            throw new Error('Invalid response data received');
        }
        
        // Update all dashboard components
        updateDashboardValues(updatedData);
        updatePieChart(updatedData);
        
        // Show success notification
        showNotification(`Successfully tracked ${carbonValue} kg CO₂ in ${category} category and updated tracked_queries.json`, 'success');
        
        // Update the track button if we have a reference to it
        if (trackButton) {
            trackButton.innerHTML = '<i class="fas fa-check"></i> Tracked';
            trackButton.classList.add('tracked');
        }
        
    } catch (error) {
        console.error('[Tracking] Error tracking query and updating tracked_queries.json:', error);
        showNotification('Failed to track impact: ' + error.message, 'error');
        
        // Reset the button if we have a reference to it
        if (event && event.target) {
            const trackButton = event.target.closest('.track-query-btn') || event.target;
            trackButton.disabled = false;
            trackButton.innerHTML = 'Track Impact';
        }
    }
}

// Function to load data for a specific period
async function loadPeriodData(period) {
    try {
        console.log(`[Tracking] Loading data for period ${period} from tracked_queries.json`);
        
        // Fetch data for the selected period from the API which loads from tracked_queries.json
        // Add cache busting parameter to ensure we're getting fresh data
        const response = await fetch(`/api/period-data?period=${period}&_t=${Date.now()}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Failed to load period data: ${response.status} ${response.statusText}`);
        }
        
        const periodData = await response.json();
        console.log('[Tracking] Received period data from tracked_queries.json:', periodData);
        
        // Validate that periodData contains what we expect
        if (!periodData || typeof periodData !== 'object') {
            console.error('[Tracking] Invalid period data received:', periodData);
            showNotification('Error: Invalid data format received', 'error');
            return;
        }
        
        // Store the selected period
        currentPeriod = period;
        
        // Update the UI with the received data
        updateDashboardValues(periodData);
        updatePieChart(periodData);
        
        // Show notification
        showNotification(`Switched to ${period} view using tracked_queries.json data`, 'success');
        
    } catch (error) {
        console.error(`[Tracking] Error loading data for period ${period} from tracked_queries.json:`, error);
        showNotification(`Failed to load ${period} data: ${error.message}`, 'error');
    }
}

// Function to explicitly refresh data from tracked_queries.json
async function refreshData() {
    try {
        // Get the refresh button
        const refreshBtn = document.getElementById('refresh-data-btn');
        if (refreshBtn) {
            refreshBtn.classList.add('loading');
        }
        
        console.log(`[Tracking] Explicitly refreshing data for period ${currentPeriod} from tracked_queries.json`);
        
        // Fetch latest data from tracked_queries.json via the API
        const response = await fetch(`/api/period-data?period=${currentPeriod}&_t=${Date.now()}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Failed to refresh data: ${response.status} ${response.statusText}`);
        }
        
        const freshData = await response.json();
        console.log('[Tracking] Received fresh data from tracked_queries.json:', freshData);
        
        // Validate that freshData contains what we expect
        if (!freshData || typeof freshData !== 'object') {
            console.error('[Tracking] Invalid refresh data received:', freshData);
            showNotification('Error: Invalid data format received', 'error');
            return;
        }
        
        // Update all dashboard components
        updateDashboardValues(freshData);
        updatePieChart(freshData);
        
        // Show success notification
        showNotification(`Refreshed dashboard with latest data from tracked_queries.json`, 'success');
        
    } catch (error) {
        console.error('[Tracking] Error refreshing data from tracked_queries.json:', error);
        showNotification('Failed to refresh data: ' + error.message, 'error');
    } finally {
        // Remove loading class from button
        const refreshBtn = document.getElementById('refresh-data-btn');
        if (refreshBtn) {
            refreshBtn.classList.remove('loading');
        }
    }
}

// Add notification function
function showNotification(message, type = 'success') {
    console.log(`Showing notification: ${message} (${type})`);
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    // Remove notification after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Debug function to help verify data
function debugTrackedData() {
    console.log('[Tracking] Debugging tracked data');
    
    // Fetch data for each period directly from tracked_queries.json
    ['daily', 'weekly', 'monthly'].forEach(async (period) => {
        try {
            const response = await fetch(`/api/period-data?period=${period}&_t=${Date.now()}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache, no-store, must-revalidate'
                }
            });
            
            if (!response.ok) {
                console.error(`[Tracking] Debug fetch failed for ${period}: ${response.status} ${response.statusText}`);
                return;
            }
            
            const data = await response.json();
            console.log(`[Tracking] Data for ${period} from tracked_queries.json:`, data);
            
            // Display key values to help debugging
            console.log(`[Tracking] ${period.toUpperCase()} VALUES:`);
            console.log(`- Total: ${data.total_carbon} kg CO₂`);
            console.log(`- Food: ${data.food_carbon} kg CO₂`);
            console.log(`- Energy: ${data.household_carbon} kg CO₂`);
            console.log(`- Mobility: ${data.transportation_carbon} kg CO₂`);
            console.log(`- Purchases: ${data.goods_carbon} kg CO₂`);
            console.log(`- Misc: ${data.misc_carbon} kg CO₂`);
            
        } catch (error) {
            console.error(`[Tracking] Debug error for ${period}:`, error);
        }
    });
}

// Initialize tracking functionality when document is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('[Tracking] Initializing tracking functionality - this will override dashboard.js period selector');
    
    // Run debug at startup to verify data
    debugTrackedData();
    
    // Store the pie chart instance when it's created
    const chartCanvas = document.getElementById('category-pie-chart');
    if (chartCanvas) {
        console.log('[Tracking] Found category pie chart canvas');
        
        // Wait for the chart to be created by dashboard.js before trying to reference it
        setTimeout(() => {
            try {
                // Try to get chart instance from Chart.js registry
                if (window.Chart && Chart.getChart) {
                    console.log('[Tracking] Getting chart from Chart.js registry');
                    categoryPieChart = Chart.getChart('category-pie-chart');
                    
                    if (categoryPieChart) {
                        console.log('[Tracking] Successfully got chart reference from registry');
                    } else if (window.categoryPieChart) {
                        console.log('[Tracking] Using global window.categoryPieChart reference');
                        categoryPieChart = window.categoryPieChart;
                    } else {
                        console.log('[Tracking] No chart reference found yet, will retry');
                        
                        // Try again after dashboard.js has had more time to initialize
                        setTimeout(() => {
                            categoryPieChart = Chart.getChart('category-pie-chart') || window.categoryPieChart;
                            console.log('[Tracking] Second attempt to get chart reference:', 
                                        categoryPieChart ? 'Success' : 'Failed');
                        }, 500);
                    }
                }
            } catch (e) {
                console.log('[Tracking] Could not get chart from Chart.js registry:', e);
            }
        }, 100);
    }
    
    // Add event listeners to period selector
    const periodSelector = document.getElementById('time-period-selector');
    if (periodSelector) {
        // Remove any existing event listeners that might have been set by dashboard.js
        const newPeriodSelector = periodSelector.cloneNode(true);
        periodSelector.parentNode.replaceChild(newPeriodSelector, periodSelector);
        
        // Add our event listener to the fresh element
        newPeriodSelector.addEventListener('change', function() {
            const selectedPeriod = this.value;
            console.log(`[Tracking] Period changed to: ${selectedPeriod}`);
            
            // Force fetch from tracked_queries.json instead of using any cached data
            loadPeriodData(selectedPeriod);
        });
        
        // Initialize with the default selected period
        currentPeriod = newPeriodSelector.value;
        console.log(`[Tracking] Initial period set to: ${currentPeriod}`);
        
        // Load initial data explicitly from tracked_queries.json
        // This ensures we have the correct data right at the start
        setTimeout(function() {
            loadPeriodData(currentPeriod);
        }, 200);
    }
    
    // Add event listener to refresh button
    const refreshBtn = document.getElementById('refresh-data-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            refreshData();
        });
        console.log('[Tracking] Added event listener to refresh button');
    }
}); 