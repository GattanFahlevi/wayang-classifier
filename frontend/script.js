class WayangClassifierApp {
    constructor() {
        // Nanti ganti dengan URL Render kamu
        this.API_BASE_URL = 'https://your-backend-url.onrender.com';
        this.initializeEventListeners();
    }
    
    // ... rest of your existing code
    
    async predictImage() {
        // ... your existing code ...
        try {
            const response = await fetch(`${this.API_BASE_URL}/predict`, {
                method: 'POST',
                body: formData
            });
            // ... rest of your code ...
        } catch (error) {
            console.error('Error:', error);
            alert('Error: ' + error.message);
        }
    }
}