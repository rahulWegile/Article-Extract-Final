import { useEffect, useState } from "react";

import api from "../api/api";

import Navbar from "../components/Navbar";
import UploadCard from "../components/UploadCard";
import DocumentCard from "../components/DocumentCard";

function Home() {

    const [documents, setDocuments] = useState([]);

    const loadDocuments = async () => {

        try {

            const response = await api.get("/documents");

            setDocuments(response.data);

        }

        catch (error) {

            console.error(error);

        }

    };

    useEffect(() => {

        loadDocuments();

    }, []);

    const handleDocumentDeleted = (deletedDocumentId) => {

        setDocuments(

            (previous) => previous.filter(

                (document) => document.document_id !== deletedDocumentId

            )

        );

    };

    return (

        <div>

            <Navbar />

            <main className="container">

                <UploadCard

                    onUploadSuccess={loadDocuments}

                />

                <section className="documents">

                    <h2>

                        Processed Documents

                    </h2>

                    {

                        documents.length === 0

                        ?

                        (

                            <p>

                                No documents found.

                            </p>

                        )

                        :

                        (

                            documents.map(

                                (document) => (

                                    <DocumentCard

                                        key={document.document_id}

                                        document={document}

                                        onDeleted={handleDocumentDeleted}

                                    />

                                )

                            )

                        )

                    }

                </section>

            </main>

        </div>

    );

}

export default Home;