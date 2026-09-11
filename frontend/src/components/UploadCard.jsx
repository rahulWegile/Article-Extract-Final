import { useRef, useState } from "react";
import api from "../api/api";

// Temporarily disabled -- flip back to true to show the real
// stage/percentage from /upload/status polling instead of a plain
// indeterminate bar. Polling itself stays on regardless (the backend
// only reports completion that way now), this just hides the numbers.
const SHOW_LIVE_PROGRESS = false;

function formatBytes(bytes) {

    if (!bytes && bytes !== 0) {
        return "";
    }

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    const kb = bytes / 1024;

    if (kb < 1024) {
        return `${kb.toFixed(kb < 10 ? 1 : 0)} KB`;
    }

    const mb = kb / 1024;

    return `${mb.toFixed(mb < 10 ? 1 : 0)} MB`;

}

function UploadCard({ onUploadSuccess, onNotify, onBoundariesCreated }) {

    const fileInput = useRef(null);

    const [file, setFile] = useState(null);

    const [uploading, setUploading] = useState(false);

    const [progress, setProgress] = useState(0);

    const [stage, setStage] = useState("");

    const [isDragOver, setIsDragOver] = useState(false);

    const chooseFile = () => {

        fileInput.current.click();

    };

    const acceptFile = (candidate) => {

        if (!candidate) {
            return;
        }

        if (candidate.type !== "application/pdf" && !candidate.name?.toLowerCase().endsWith(".pdf")) {

            onNotify?.("error", "Only PDF files are supported.");
            return;

        }

        setFile(candidate);

    };

    const onFileChange = (event) => {

        if (event.target.files.length > 0) {

            acceptFile(event.target.files[0]);

        }

    };

    const onDrop = (event) => {

        event.preventDefault();
        setIsDragOver(false);

        if (uploading) {
            return;
        }

        if (event.dataTransfer.files.length > 0) {

            acceptFile(event.dataTransfer.files[0]);

        }

    };

    const clearFile = (event) => {

        event.stopPropagation();

        setFile(null);

        if (fileInput.current) {
            fileInput.current.value = "";
        }

    };

    const uploadPDF = async () => {

        if (!file) {

            onNotify?.("error", "Please select a PDF to upload.");
            return;

        }

        const formData = new FormData();

        formData.append("file", file);

        let pollTimer = null;
        let boundariesNotified = false;

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

                        if (
                            job.event === "boundaries_created" &&
                            !boundariesNotified
                        ) {

                            boundariesNotified = true;

                            onBoundariesCreated?.(job.article_count ?? null);

                        }

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

            onNotify?.("success", "Upload completed successfully.");

            setFile(null);

            fileInput.current.value = "";

            if (onUploadSuccess) {

                onUploadSuccess();

            }

        } catch (error) {

            console.error(error);

            onNotify?.("error", `Upload failed: ${error.message || error}`);

            if (onUploadSuccess) {
                onUploadSuccess();
            }

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

            <div className="upload-card-header">

                <h2>Upload Newspaper PDF</h2>

                <p>Add a scanned edition to detect article boundaries automatically.</p>

            </div>

            <input

                ref={fileInput}

                type="file"

                accept=".pdf,application/pdf"

                onChange={onFileChange}

                style={{ display: "none" }}

            />

            <div
                className={
                    "upload-dropzone" +
                    (isDragOver ? " is-dragover" : "") +
                    (uploading ? " is-disabled" : "")
                }
                onClick={uploading ? undefined : chooseFile}
                onDragOver={(event) => {
                    event.preventDefault();
                    if (!uploading) {
                        setIsDragOver(true);
                    }
                }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={onDrop}
                role="button"
                tabIndex={0}
                aria-disabled={uploading}
            >

                {file ? (

                    <div className="upload-file-chip">

                        <span className="upload-file-icon" aria-hidden="true">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path
                                    d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinejoin="round"
                                />
                                <path d="M14 3v5h5" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
                            </svg>
                        </span>

                        <span className="upload-file-info">
                            <span className="upload-file-name">{file.name}</span>
                            <span className="upload-file-size">{formatBytes(file.size)}</span>
                        </span>

                        {!uploading && (

                            <button
                                type="button"
                                className="upload-file-remove"
                                onClick={clearFile}
                                aria-label="Remove selected file"
                            >
                                ×
                            </button>

                        )}

                    </div>

                ) : (

                    <div className="upload-dropzone-empty">

                        <span className="upload-dropzone-icon" aria-hidden="true">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path
                                    d="M12 16V4m0 0 4 4m-4-4-4 4"
                                    stroke="currentColor"
                                    strokeWidth="1.75"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                />
                                <path
                                    d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"
                                    stroke="currentColor"
                                    strokeWidth="1.75"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                />
                            </svg>
                        </span>

                        <p>
                            <strong>Drag &amp; drop</strong> a PDF here, or click to browse
                        </p>

                        <span className="upload-dropzone-hint">Supports single PDF files</span>

                    </div>

                )}

            </div>

            <div className="upload-actions">

                <button

                    type="button"
                    className="btn btn-secondary"
                    onClick={chooseFile}
                    disabled={uploading}

                >

                    Choose PDF

                </button>

                <button

                    type="button"
                    className="btn btn-primary"
                    onClick={uploadPDF}
                    disabled={uploading || !file}

                >

                    {uploading ? "Uploading…" : "Upload"}

                </button>

            </div>

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
