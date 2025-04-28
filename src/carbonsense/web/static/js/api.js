// CarbonSense API Service
// Handles all API interactions for the CarbonSense application

console.log('[API] Initializing CarbonSense API Service');

// Create a namespace for our API 
window.CarbonSenseAPI = (function() {
    console.log('[API] Building CarbonSenseAPI object');
    
    // Base URL for API endpoints (in case we need to change it later)
    const baseURL = '';
    console.log('[API] Base URL configured as:', baseURL || '(current domain)');
    
    // Error handling for fetch requests
    async function fetchWithErrorHandling(url, options = {}) {
        console.log('[API] Request:', options.method || 'GET', url);
        console.log('[API] Request options:', options);
        
        try {
            const response = await fetch(url, options);
            console.log('[API] Response status:', response.status, response.statusText);
            
            if (!response.ok) {
                console.error('[API] Error response:', response.status, response.statusText, 'for URL:', url);
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('[API] Response data:', data);
            return data;
        } catch (error) {
            console.error('[API] Fetch error for URL:', url, 'Error:', error.message);
            throw error;
        }
    }
    
    // Query carbon footprint for a text query
    async function queryCarbonFootprint(query) {
        console.log('[API] Querying carbon footprint for:', query);
        
        if (!query) {
            console.error('[API] Query is required for carbon footprint calculation');
            throw new Error('Query is required');
        }
        
        const url = `${baseURL}/api/query`;
        
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        };
        
        try {
            const result = await fetchWithErrorHandling(url, options);
            console.log('[API] Carbon footprint query successful');
            return result;
        } catch (error) {
            console.error('[API] Carbon footprint query failed:', error);
            throw error;
        }
    }
    
    // Fetch dashboard data
    async function fetchDashboardData() {
        console.log('[API] Fetching dashboard data');
        
        const url = `${baseURL}/api/dashboard-data`;
        
        try {
            const result = await fetchWithErrorHandling(url);
            console.log('[API] Dashboard data fetch successful');
            return result;
        } catch (error) {
            console.error('[API] Dashboard data fetch failed:', error);
            throw error;
        }
    }
    
    // Log user activity
    async function logActivity(activity) {
        console.log('[API] Logging activity:', activity);
        
        if (!activity) {
            console.error('[API] Activity is required');
            throw new Error('Activity is required');
        }
        
        const url = `${baseURL}/api/activity`;
        
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ activity })
        };
        
        try {
            const result = await fetchWithErrorHandling(url, options);
            console.log('[API] Activity logging successful');
            return result;
        } catch (error) {
            console.error('[API] Activity logging failed:', error);
            throw error;
        }
    }
    
    // Get personalized tips
    async function getPersonalizedTips(category = null) {
        console.log('[API] Getting personalized tips', category ? `for category: ${category}` : '(all categories)');
        
        let url = `${baseURL}/api/tips`;
        
        if (category) {
            url += `?category=${encodeURIComponent(category)}`;
        }
        
        try {
            const result = await fetchWithErrorHandling(url);
            console.log('[API] Retrieved personalized tips successfully, count:', result.tips?.length || 0);
            return result;
        } catch (error) {
            console.error('[API] Failed to get personalized tips:', error);
            throw error;
        }
    }
    
    // Apply a carbon-saving tip
    async function applyTip(tipId) {
        console.log('[API] Applying tip ID:', tipId);
        
        if (!tipId) {
            console.error('[API] Tip ID is required');
            throw new Error('Tip ID is required');
        }
        
        const url = `${baseURL}/api/tips/apply`;
        
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ tip_id: tipId })
        };
        
        try {
            const result = await fetchWithErrorHandling(url, options);
            console.log('[API] Tip applied successfully, result:', result);
            return result;
        } catch (error) {
            console.error('[API] Failed to apply tip:', error);
            throw error;
        }
    }
    
    // Compare products carbon footprint
    async function compareProducts(products) {
        console.log('[API] Comparing products:', products);
        
        if (!products || !Array.isArray(products) || products.length < 2) {
            console.error('[API] At least two products are required for comparison');
            throw new Error('At least two products are required for comparison');
        }
        
        const url = `${baseURL}/api/compare-products`;
        
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ products })
        };
        
        try {
            const result = await fetchWithErrorHandling(url, options);
            console.log('[API] Product comparison successful');
            return result;
        } catch (error) {
            console.error('[API] Product comparison failed:', error);
            throw error;
        }
    }
    
    // Create API object with all functions
    const api = {
        fetchDashboardData,
        queryCarbonFootprint,
        logActivity,
        getPersonalizedTips,
        applyTip,
        compareProducts
    };
    
    console.log('[API] API object created with these methods:', Object.keys(api).join(', '));
    
    // Public API
    return api;
})();

// Log that we've initialized the API object
console.log('[API] CarbonSenseAPI initialized with these methods:', 
    Object.keys(window.CarbonSenseAPI || {}).join(', '));

// Initialize API service without console logging
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        console.log('[API] CarbonSenseAPI loaded and ready (DOMContentLoaded)');
        console.log('[API] Available methods:', Object.keys(window.CarbonSenseAPI || {}).join(', '));
    });
} else {
    console.log('[API] CarbonSenseAPI loaded and ready (immediate)');
    console.log('[API] Available methods:', Object.keys(window.CarbonSenseAPI || {}).join(', '));
} 