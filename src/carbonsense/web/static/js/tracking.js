// Global variable to store the pie chart instance
let categoryPieChart = null;

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
                '#4CAF50', // Food - matching dashboard.js colors
                '#2196F3', // Energy
                '#9C27B0', // Mobility
                '#FFC107', // Purchases
                '#9E9E9E'  // Misc
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
    const targetValue = 12.15; // This should match your target value
    const totalCarbon = updatedData.total_carbon;
    const changeElement = document.querySelector('#carbon-total-box .stat-change');
    if (changeElement) {
        const percentChange = ((Math.abs(totalCarbon - targetValue) / targetValue) * 100).toFixed(1);
        changeElement.textContent = totalCarbon > targetValue ? `+${percentChange}%` : `-${percentChange}%`;
        changeElement.className = `stat-change ${totalCarbon > targetValue ? 'negative' : 'positive'}`;
    }
}

// Enhanced trackQuery function
async function trackQuery(query, category, carbonValue, event) {
    try {
        console.log(`Tracking query in category: ${category}, with carbon value: ${carbonValue}`);
        
        // Validate the carbon value
        if (isNaN(carbonValue) || carbonValue <= 0) {
            console.error(`Invalid carbon value: ${carbonValue}`);
            showNotification('Invalid carbon value for tracking', 'error');
            return;
        }
        
        // Store button reference before fetch if event exists
        let trackButton = null;
        if (event && event.target) {
            trackButton = event.target.closest('.track-query-btn') || event.target;
            console.log('Found track button:', trackButton);
            
            // Disable button immediately to prevent multiple clicks
            trackButton.disabled = true;
            trackButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Tracking...';
        } else {
            console.log('No event or event.target provided');
        }
        
        const response = await fetch('/api/track-query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                category: category,
                carbon_value: carbonValue
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to track query: ${response.status} ${response.statusText}`);
        }
        
        const updatedData = await response.json();
        console.log('Received updated data:', updatedData);
        
        // Validate received data
        if (!updatedData || typeof updatedData !== 'object') {
            throw new Error('Invalid response data received');
        }
        
        // Update all dashboard components
        updateDashboardValues(updatedData);
        updatePieChart(updatedData);
        
        // Show success notification
        showNotification(`Successfully tracked ${carbonValue} kg CO₂ in ${category} category`, 'success');
        
        // Update the track button if we have a reference to it
        if (trackButton) {
            trackButton.innerHTML = '<i class="fas fa-check"></i> Tracked';
            trackButton.classList.add('tracked');
        }
        
    } catch (error) {
        console.error('Error tracking query:', error);
        showNotification('Failed to track impact: ' + error.message, 'error');
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

// Initialize tracking functionality when document is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing tracking functionality');
    
    // Store the pie chart instance when it's created
    const chartCanvas = document.getElementById('category-pie-chart');
    if (chartCanvas) {
        console.log('Found category pie chart canvas');
        
        // Try to get the chart instance several ways
        if (window.categoryPieChart) {
            console.log('Using window.categoryPieChart');
            categoryPieChart = window.categoryPieChart;
        } else {
            // Try to get chart instance from Chart.js registry
            try {
                if (window.Chart && Chart.getChart) {
                    console.log('Getting chart from Chart.js registry');
                    categoryPieChart = Chart.getChart('category-pie-chart');
                }
            } catch (e) {
                console.log('Could not get chart from Chart.js registry:', e);
            }
        }
        
        if (categoryPieChart) {
            console.log('Successfully initialized chart reference');
        } else {
            console.log('No chart reference found, will attempt to use window.categoryPieChart dynamically');
        }
    }
}); 