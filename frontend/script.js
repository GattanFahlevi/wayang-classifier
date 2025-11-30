// frontend/script.js
class WayangClassifierApp {
    constructor() {
        // GANTI DENGAN URL BACKEND KAMU SETELAH DEPLOY
        this.API_BASE_URL = 'https://your-backend-url.railway.app'; // ← GANTI INI!
        this.currentFile = null;
        this.initializeEventListeners();
        this.checkAPIHealth();
    }

    initializeEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        const predictBtn = document.getElementById('predictBtn');

        // Click event
        uploadArea.addEventListener('click', () => {
            imageInput.click();
        });

        // Drag and drop events
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleImageSelection(files[0]);
            }
        });

        // File input change
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageSelection(e.target.files[0]);
            }
        });

        // Predict button
        predictBtn.addEventListener('click', () => {
            this.predictImage();
        });
    }

    handleImageSelection(file) {
        if (!file.type.match('image.*')) {
            alert('Silakan pilih file gambar (JPG, PNG)');
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            alert('Ukuran file maksimal 5MB');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('previewImage');
            const uploadPlaceholder = document.querySelector('.upload-placeholder');
            const predictBtn = document.getElementById('predictBtn');

            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
            predictBtn.disabled = false;

            // Hide previous results
            document.getElementById('resultSection').style.display = 'none';
        };
        reader.readAsDataURL(file);
        this.currentFile = file;
    }

    async predictImage() {
        if (!this.currentFile) return;

        const loading = document.getElementById('loading');
        const predictBtn = document.getElementById('predictBtn');

        loading.style.display = 'block';
        predictBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('image', this.currentFile);

            console.log('🔄 Mengirim gambar ke API...');
            
            const response = await fetch(`${this.API_BASE_URL}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();

            if (result.success) {
                this.displayResult(result);
            } else {
                throw new Error(result.error || 'Prediksi gagal');
            }

        } catch (error) {
            console.error('Error:', error);
            this.displayError('Gagal melakukan prediksi: ' + error.message);
        } finally {
            loading.style.display = 'none';
            predictBtn.disabled = false;
        }
    }

    displayResult(result) {
        const resultSection = document.getElementById('resultSection');
        const wayangName = document.getElementById('wayangName');
        const confidence = document.getElementById('confidence');
        const probabilities = document.getElementById('probabilities');

        // Update main prediction
        wayangName.textContent = result.prediction;
        confidence.textContent = `${result.confidence.toFixed(2)}%`;

        // Add confidence color
        if (result.confidence >= 80) {
            confidence.style.color = '#22c55e'; // Green
        } else if (result.confidence >= 60) {
            confidence.style.color = '#eab308'; // Yellow
        } else {
            confidence.style.color = '#ef4444'; // Red
        }

        // Display probabilities
        probabilities.innerHTML = '';
        Object.entries(result.all_predictions).forEach(([wayang, prob]) => {
            const percentage = (prob * 100).toFixed(2);
            const probabilityBar = document.createElement('div');
            probabilityBar.className = 'probability-bar';
            
            // Highlight the predicted class
            const isPredicted = wayang === result.prediction;
            const barColor = isPredicted ? '#667eea' : '#94a3b8';
            
            probabilityBar.innerHTML = `
                <div class="probability-label">
                    <span style="font-weight: ${isPredicted ? 'bold' : 'normal'}">${wayang}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar" style="width: ${percentage}%; background: ${barColor}"></div>
                </div>
            `;
            probabilities.appendChild(probabilityBar);
        });

        resultSection.style.display = 'block';
        
        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    displayError(message) {
        const resultSection = document.getElementById('resultSection');
        const wayangName = document.getElementById('wayangName');
        const confidence = document.getElementById('confidence');
        const probabilities = document.getElementById('probabilities');

        wayangName.textContent = 'ERROR';
        confidence.textContent = '0%';
        confidence.style.color = '#ef4444';
        
        probabilities.innerHTML = `
            <div style="text-align: center; color: #ef4444; padding: 20px;">
                <p>${message}</p>
                <p>Pastikan backend API sedang berjalan</p>
            </div>
        `;

        resultSection.style.display = 'block';
    }

    async checkAPIHealth() {
        try {
            const response = await fetch(`${this.API_BASE_URL}/health`);
            const data = await response.json();
            
            const statusIndicator = document.createElement('div');
            statusIndicator.style.cssText = `
                position: fixed;
                top: 10px;
                right: 10px;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                z-index: 1000;
            `;
            
            if (data.status === 'healthy' && data.model_loaded) {
                statusIndicator.style.background = '#22c55e';
                statusIndicator.style.color = 'white';
                statusIndicator.textContent = 'API ✅';
                console.log('✅ API Connected:', data);
            } else {
                statusIndicator.style.background = '#eab308';
                statusIndicator.style.color = 'black';
                statusIndicator.textContent = 'API ⚠️';
                console.warn('⚠️ API Warning:', data);
            }
            
            document.body.appendChild(statusIndicator);
            
        } catch (error) {
            console.error('❌ API Health Check Failed:', error);
            
            const statusIndicator = document.createElement('div');
            statusIndicator.style.cssText = `
                position: fixed;
                top: 10px;
                right: 10px;
                padding: 5px 10px;
                background: #ef4444;
                color: white;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                z-index: 1000;
            `;
            statusIndicator.textContent = 'API ❌';
            document.body.appendChild(statusIndicator);
        }
    }
}

// Wayang information with images (optional)
const wayangInfo = {
    'GARENG': {
        description: 'Wayang Gareng dikenal sebagai punakawan yang bijaksana dan sederhana.',
        image: 'https://via.placeholder.com/100?text=GARENG'
    },
    'SEMAR': {
        description: 'Wayang Semar adalah punakawan tertua dan paling bijaksana.',
        image: 'https://via.placeholder.com/100?text=SEMAR'
    },
    'PETRUK': {
        description: 'Wayang Petruk dikenal dengan hidungnya yang panjang dan lucu.',
        image: 'https://via.placeholder.com/100?text=PETRUK'
    },
    'BAGONG': {
        description: 'Wayang Bagong adalah punakawan yang jenaka dan cerdik.',
        image: 'https://via.placeholder.com/100?text=BAGONG'
    }
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new WayangClassifierApp();
    
    // Add some interactive features
    const wayangItems = document.querySelectorAll('.wayang-item');
    wayangItems.forEach(item => {
        item.addEventListener('mouseenter', () => {
            item.style.transform = 'scale(1.05)';
            item.style.transition = 'transform 0.2s ease';
        });
        
        item.addEventListener('mouseleave', () => {
            item.style.transform = 'scale(1)';
        });
    });
});

// Utility function for API testing (development only)
window.testAPI = function() {
    const app = new WayangClassifierApp();
    app.checkAPIHealth();
};
