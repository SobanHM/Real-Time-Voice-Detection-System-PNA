function predict() {

    const fileInput = document.getElementById("audioInput");

    if (!fileInput.files.length) {
        alert("Upload a WAV file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {

        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById("speaker").innerText = data.speaker;
        document.getElementById("confidence").innerText =
            (data.confidence * 100).toFixed(2) + "%";

        document.getElementById("resultBox").classList.remove("hidden");
    })
    .catch(err => alert("Server error"));
}
