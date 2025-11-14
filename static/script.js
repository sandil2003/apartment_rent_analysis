document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('errorContainer');
    const predictedRentElement = document.getElementById('predictedRent');
    const errorMessageElement = document.getElementById('errorMessage');

    const propertiesInput = document.getElementById('properties');
    const shapeAreaInput = document.getElementById('shape_area');
    const yearInput = document.getElementById('year');

    function calculateDerivedFeatures() {
        const properties = parseFloat(propertiesInput.value) || 0;
        const shapeArea = parseFloat(shapeAreaInput.value) || 0;
        const year = parseFloat(yearInput.value) || 2024;
        
        const propertiesPerArea = properties / (shapeArea + 1);
        const yearSince2000 = year - 2000;
        
        return {
            properties_per_area: propertiesPerArea,
            year_since_2000: yearSince2000
        };
    }

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        errorContainer.classList.add('hidden');
        resultContainer.classList.add('hidden');
        
        const formData = new FormData(form);
        const derivedFeatures = calculateDerivedFeatures();
        
        const data = {
            year: formData.get('year'),
            community_id: formData.get('community_id'),
            properties: formData.get('properties'),
            shape_area: formData.get('shape_area'),
            shape_length: formData.get('shape_length'),
            mixed_rate: formData.get('mixed_rate'),
            area_encoded: formData.get('area_encoded'),
            cost_category_encoded: formData.get('cost_category_encoded'),
            change_category_encoded: formData.get('change_category_encoded'),
            properties_per_area: derivedFeatures.properties_per_area,
            year_since_2000: derivedFeatures.year_since_2000
        };

        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Predicting...';
        submitButton.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok && result.success) {
                predictedRentElement.textContent = result.formatted_rent;
                resultContainer.classList.remove('hidden');
                
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
        } catch (error) {
            errorMessageElement.textContent = error.message;
            errorContainer.classList.remove('hidden');
            errorContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } finally {
            submitButton.innerHTML = originalButtonText;
            submitButton.disabled = false;
        }
    });

    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.querySelector('label').classList.add('text-purple-600');
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.querySelector('label').classList.remove('text-purple-600');
        });
    });
});
