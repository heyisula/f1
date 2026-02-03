// F1 Predictor - Main Application Logic

// Update the grid position display when slider moves
const gridSlider = document.getElementById('grid');
const gridDisplay = document.getElementById('grid-val');

gridSlider.addEventListener('input', (e) => {
    gridDisplay.textContent = e.target.value;
});

// Load dropdown options when page loads
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/meta');
        const data = await response.json();

        fillDropdown('driver', data.drivers, 'max_verstappen');
        fillDropdown('constructor', data.constructors, 'red_bull');
        fillDropdown('circuit', data.circuits);
    } catch (error) {
        console.error("Couldn't load dropdown data:", error);
    }
});

// Convert snake_case to Title Case (e.g., max_verstappen -> Max Verstappen)
function formatName(snakeCaseStr) {
    return snakeCaseStr
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

function fillDropdown(dropdownId, items, defaultValue) {
    const dropdown = document.getElementById(dropdownId);

    // Sort alphabetically for easier browsing
    items.sort();

    items.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = formatName(item); // Format the display text
        dropdown.appendChild(option);
    });

    // Set a sensible default if provided
    if (defaultValue && items.includes(defaultValue)) {
        dropdown.value = defaultValue;
    }
}

// Handle form submission
document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.querySelector('.btn-predict');
    const loader = document.getElementById('loader');
    const ring = document.getElementById('prob-ring');
    const probText = document.getElementById('prob-text');
    const label = document.getElementById('prediction-label');

    // Show loading state
    submitBtn.disabled = true;
    submitBtn.style.opacity = '0.6';
    loader.style.display = 'block';
    ring.style.display = 'none';
    label.classList.remove('visible');

    // Gather form data
    const formData = {
        driver: document.getElementById('driver').value,
        constructor: document.getElementById('constructor').value,
        circuit: document.getElementById('circuit').value,
        grid: parseInt(document.getElementById('grid').value),
        laps: parseInt(document.getElementById('laps').value),
        points: parseFloat(document.getElementById('points').value)
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        // Small delay for effect
        setTimeout(() => {
            loader.style.display = 'none';
            ring.style.display = 'flex';

            // Calculate and display percentage
            const percentage = (result.probability * 100).toFixed(1);
            probText.textContent = percentage + '%';

            // Use gradient colors for the ring based on probability
            let ringColor;
            if (result.probability >= 0.7) {
                ringColor = 'linear-gradient(135deg, var(--success-green), var(--accent-teal))';
            } else if (result.probability >= 0.4) {
                ringColor = 'linear-gradient(135deg, var(--accent-orange), var(--accent-purple))';
            } else {
                ringColor = 'rgba(139, 148, 158, 0.5)';
            }

            ring.style.background = `conic-gradient(${ringColor} ${percentage * 3.6}deg, rgba(255,255,255,0.05) ${percentage * 3.6}deg)`;

            // Update prediction text
            if (result.is_winner) {
                label.textContent = '🏆 Winner Predicted!';
                label.style.background = 'linear-gradient(135deg, var(--success-green), var(--accent-teal))';
            } else {
                label.textContent = 'Unlikely to Win';
                label.style.background = 'none';
                label.style.color = 'var(--text-muted)';
            }
            label.style.backgroundClip = 'text';
            label.style.webkitBackgroundClip = 'text';
            label.style.webkitTextFillColor = result.is_winner ? 'transparent' : 'inherit';
            label.classList.add('visible');

            submitBtn.disabled = false;
            submitBtn.style.opacity = '1';
        }, 800);

    } catch (error) {
        console.error('Prediction failed:', error);
        loader.style.display = 'none';
        ring.style.display = 'flex';
        probText.textContent = 'Error';
        submitBtn.disabled = false;
        submitBtn.style.opacity = '1';
    }
});
