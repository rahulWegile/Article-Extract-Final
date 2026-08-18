import { useRef, useState } from "react";
import api from "../api/api";

function UploadCard({ onUploadSuccess }) {

    const fileInput = useRef(null);

    const [file, setFile] = useState(null);

    const [uploading, setUploading] = useState(false);

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

        try {

            setUploading(true);

            await api.post(

                "/upload",

                formData

            );

            alert("Upload completed.");

            setFile(null);

            fileInput.current.value = "";

            if (onUploadSuccess) {

                onUploadSuccess();

            }

        } catch (error) {

            console.error(error);

            alert("Upload failed.");

        } finally {

            setUploading(false);

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

        </div>

    );

}

export default UploadCard;