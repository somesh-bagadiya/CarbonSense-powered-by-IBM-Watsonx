// CarbonSense API Service
// Handles all API interactions for the CarbonSense application

console.log('[API] Initializing CarbonSense API Service');

// Create a namespace for our API 
window.CarbonSenseAPI = (function() {
    
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
    
    // Start voice recording
    async function startVoiceRecording() {
        console.log('[API] Starting voice recording session');
        
        const url = `${baseURL}/api/start-recording`;
        
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        try {
            const result = await fetchWithErrorHandling(url, options);
            console.log('[API] Voice recording session started successfully with ID:', result.session_id);
            return result;
        } catch (error) {
            console.error('[API] Failed to start voice recording session:', error);
            throw error;
        }
    }
    
    // Stop voice recording and get results
    async function stopVoiceRecording(sessionId) {
        console.log('[API] Stopping voice recording for session:', sessionId);
        
        if (!sessionId) {
            console.error('[API] Session ID is required to stop recording');
            throw new Error('Session ID is required to stop recording');
        }
        
        const url = `${baseURL}/api/stop-recording/${sessionId}`;
        
        const options = {
            method: 'POST'
        };
        
        try {
            const result = await fetchWithErrorHandling(url, options);
            console.log('[API] Voice recording stopped successfully, result:', result);
            return result;
        } catch (error) {
            console.log('[API] Error stopping recording:', error.message);
            
            // Special handling for session not found errors
            if (error.message.includes('404')) {
                console.log('[API] 404 error detected, handling as session expired case');
                return { 
                    status: 'error', 
                    message: 'Recording session not found. It may have already completed processing.' 
                };
            }
            throw error;
        }
    }
    
    // Check processing status for voice recording
    async function checkProcessingStatus(sessionId) {
        console.log('[API] Checking processing status for session:', sessionId);
        
        if (!sessionId) {
            console.error('[API] Session ID is required to check processing status');
            throw new Error('Session ID is required to check processing status');
        }
        
        const url = `${baseURL}/api/check-processing/${sessionId}`;
        
        try {
            const result = await fetchWithErrorHandling(url);
            console.log('[API] Processing status check result:', result);
            return result;
        } catch (error) {
            console.log('[API] Error checking processing status:', error.message);
            // If we get an error checking status, return a not_found response rather than throwing
            return { status: 'not_found', message: error.message };
        }
    }
    
    // Track a carbon query
    async function trackCarbonQuery(query, category, carbonValue) {
        console.log('[API] Tracking carbon query:', { query, category, carbonValue });
        
        if (!query) {
            console.error('[API] Query is required for tracking');
            throw new Error('Query is required for tracking');
        }
        
        if (!category) {
            console.error('[API] Category is required for tracking');
            throw new Error('Category is required for tracking');
        }
        
        if (carbonValue === undefined || carbonValue === null) {
            console.error('[API] Carbon value is required for tracking');
            throw new Error('Carbon value is required for tracking');
        }
        
        const url = `${baseURL}/api/track-query`;
        
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                category,
                carbon_value: carbonValue
            })
        };
        
        try {
            const result = await fetchWithErrorHandling(url, options);
            console.log('[API] Carbon query tracked successfully, updated data:', result);
            return result;
        } catch (error) {
            console.error('[API] Failed to track carbon query:', error);
            throw error;
        }
    }
    
    // Upload voice data directly (alternative to the streaming approach)
    async function uploadVoiceData(audioBlob) {
        console.log('[API] Uploading voice data, blob size:', audioBlob?.size || 'N/A');
        
        if (!audioBlob) {
            console.error('[API] Audio data is required');
            throw new Error('Audio data is required');
        }
        
        const url = `${baseURL}/api/voice-query`;
        
        const formData = new FormData();
        formData.append('audio_data', audioBlob, 'voice_query.wav');
        
        const options = {
            method: 'POST',
            body: formData
        };
        
        try {
            const result = await fetchWithErrorHandling(url, options);
            console.log('[API] Voice data uploaded and processed successfully');
            return result;
        } catch (error) {
            console.error('[API] Failed to upload voice data:', error);
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
    
    // Public API
    return {
        queryCarbonFootprint,
        startVoiceRecording,
        stopVoiceRecording,
        checkProcessingStatus,
        trackCarbonQuery,
        uploadVoiceData,
        getPersonalizedTips,
        applyTip
    };
})();

// Initialize API service without console logging
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        console.log('[API] CarbonSenseAPI loaded and ready (DOMContentLoaded)');
    });
} else {
    console.log('[API] CarbonSenseAPI loaded and ready (immediate)');
} 