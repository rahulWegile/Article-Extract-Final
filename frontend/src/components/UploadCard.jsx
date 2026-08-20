import { useRef, useState } from "react";
import api from "../api/api";

// Temporarily disabled -- flip back to true to show the real
// stage/percentage from /upload/status polling instead of a plain
// indeterminate bar. Polling itself stays on regardless (the backend
// only reports completion that way now), this just hides the numbers.
const SHOW_LIVE_PROGRESS = false;

function UploadCard({ onUploadSuccess }) {

    const fileInput = useRef(null);

    const [file, setFile] = useState(null);

    const [uploading, setUploading] = useState(false);

    const [progress, setProgress] = useState(0);

    const [stage, setStage] = useState("");

    const chooseFile = () => {

        fileInput.current.click();

    };

    const onFileChange = (event) => {

        if (event.target.files.length > 0) {

            setFile(event.target.files[0]);

        }

    };

    const uploadPDF = async () => {

        if (!file) {

            alert("Please select a PDF.");

            return;

        }

        const formData = new FormData();

        formData.append("file", file);

        let pollTimer = null;

        try {

            setUploading(true);

            setProgress(0);

            setStage("Uploading...");

            const response = await api.post(

                "/upload",

                formData,

                {
                    onUploadProgress: (event) => {

                        if (!event.total) {

                            return;

                        }

                        setProgress(
                            Math.round((event.loaded * 100) / event.total)
                        );

                    },
                }

            );

            const jobId = response.data.job_id;

            setStage("Starting...");

            await new Promise((resolve, reject) => {

                pollTimer = setInterval(async () => {

                    try {

                        const statusResponse = await api.get(

                            `/upload/status/${jobId}`

                        );

                        const job = statusResponse.data;

                        setProgress(job.progress ?? 0);

                        setStage(job.stage ?? "");

                        if (job.status === "completed") {

                            clearInterval(pollTimer);

                            resolve();

                        } else if (job.status === "failed") {

                            clearInterval(pollTimer);

                            reject(new Error(job.error || "Pipeline failed."));

                        }

                    } catch (pollError) {

                        clearInterval(pollTimer);

                        reject(pollError);

                    }

                }, 4000);

            });

            alert("Upload completed.");

            setFile(null);

            fileInput.current.value = "";

            if (onUploadSuccess) {

                onUploadSuccess();

            }

        } catch (error) {

            console.error(error);

            alert(`Upload failed: ${error.message || error}`);

        } finally {

            if (pollTimer) {

                clearInterval(pollTimer);

            }

            setUploading(false);

            setProgress(0);

            setStage("");

        }

    };

    return (

        <div className="upload-card">

            <h2>

                Upload Newspaper PDF

            </h2>

            <input

                ref={fileInput}

                type="file"

                accept=".pdf"

                onChange={onFileChange}

                style={{ display: "none" }}

            />

            <div className="selected-file">

                {file ? file.name : "No file selected"}

            </div>

            <button

                className="choose-btn"

                onClick={chooseFile}

            >

                Choose PDF

            </button>

            <button

                className="upload-btn"

                onClick={uploadPDF}

                disabled={uploading}

            >

                {uploading ? "Uploading..." : "Upload"}

            </button>

            {uploading && SHOW_LIVE_PROGRESS && (

                <>

                    <div className="upload-progress">

                        <div

                            className="upload-progress-fill"

                            style={{ width: `${progress}%` }}

                        />

                    </div>

                    <div className="upload-stage">

                        {stage} {stage ? `(${progress}%)` : ""}

                    </div>

                </>

            )}

            {uploading && !SHOW_LIVE_PROGRESS && (

                <div className="upload-progress">

                    <div className="upload-progress-fill upload-progress-indeterminate" />

                </div>

            )}

        </div>

    );

}

export default UploadCard;