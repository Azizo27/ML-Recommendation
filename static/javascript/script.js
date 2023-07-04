function checkResult() {
    fetch('/check_result')
        .then(response => response.json())
        .then(data => {
            if (data.result_available) {
                window.location.href = '/result';  // Redirect to the result page
            } else {
                setTimeout(checkResult, 1000);  // Check again after 1 second
            }
        });
}

// Start checking for the result when the page loads
window.onload = checkResult;