import React, { useState } from "react";
import { auth } from "../firebase-config";

function FileUpload() {
    const [selectedFile1, setSelectedFile1] = useState(null);
    const [selectedFile2, setSelectedFile2] = useState(null);
    const [mode, setMode] = useState("");
    const [backendResponse, setBackendResponse] = useState("");

    const handleFileChange1 = (event) => {
        setSelectedFile1(event.target.files[0]);
    };

    const handleFileChange2 = (event) => {
        setSelectedFile2(event.target.files[0]);
    };

    const handleModeChange = (event) => {
        setMode(event.target.value);
    };

    const handleUpload = async () => {
        if (!selectedFile1 || !selectedFile2) {
            alert("Please select both files!");
            return;
        }

        if (mode !== "1" && mode !== "2") {
            alert("Please select a mode (1 or 2)!");
            return;
        }

        const formData = new FormData();
        formData.append("file1", selectedFile1);
        formData.append("file2", selectedFile2);
        formData.append("mode", mode);
        formData.append("userId", auth.currentUser.uid);

        try {
            const response = await fetch("http://localhost:5000/upload", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            setBackendResponse(JSON.stringify(data, null, 2)); // Store the response data in state
        } catch (error) {
            console.error("Error uploading file:", error);
            setBackendResponse(`Error: ${error.message}`); // Store error message in state
        }
    };

    return (
        <div>
            <input type='file' onChange={handleFileChange1} />
            <input type='file' onChange={handleFileChange2} />
            <div>
                <label>
                    Choose batches selection mode (1 or 2):
                    <select value={mode} onChange={handleModeChange}>
                        <option value=''>Select Mode</option>
                        <option value='1'>1</option>
                        <option value='2'>2</option>
                    </select>
                </label>
            </div>
            <button onClick={handleUpload}>Upload</button>
            <div>
                <h3>Backend Response:</h3>
                <pre>{backendResponse}</pre>
            </div>
        </div>
    );
}

export default FileUpload;
