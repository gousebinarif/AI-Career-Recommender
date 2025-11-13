document.getElementById("career-form").addEventListener("submit", async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());
    const spinner = document.getElementById("loading-spinner");
    const resultsDiv = document.getElementById("results");
    const ctx = document.getElementById("resultChart").getContext("2d");

    spinner.classList.remove("hidden");
    resultsDiv.innerHTML = "";
    if (window.resultChartInstance) {
        window.resultChartInstance.destroy();
    }

    const response = await fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();
    spinner.classList.add("hidden");

    resultsDiv.innerHTML = result.recommendations.map((r, i) => `<p>${i + 1}. ${r}</p>`).join("");

    window.resultChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: result.recommendations,
            datasets: [{
                label: "Match Confidence (%)",
                data: result.scores,
                backgroundColor: "rgba(0, 180, 255, 0.7)",
                borderColor: "#00bfff",
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: "#b0dfff" }
                },
                x: {
                    ticks: { color: "#b0dfff" }
                }
            },
            plugins: {
                legend: { labels: { color: "#b0dfff" } }
            }
        }
    });
});
