import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import FileUpload from "./Components/FileUpload";
import OldRuns from "./Components/OldRuns";
import Register from "./Components/Register";
import Login from "./Components/Login";
import Dashboard from "./Components/Dashboard";
import LandingPage from "./Components/LandingPage";
import { auth } from "./firebase-config";

function App() {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        console.log("Setting up auth state changed listener");
        const unsubscribe = auth.onAuthStateChanged((user) => {
            console.log("Auth state changed, user:", user);
            setCurrentUser(user);
            setLoading(false);
        });

        return () => {
            console.log("Unsubscribing auth listener");
            unsubscribe();
        };
    }, []);

    const manualSetCurrentUser = (user) => {
        setCurrentUser(user);
    };

    if (loading) {
        return <div>Loading...</div>;
    }

    return (
        <Router>
            <div>
                <nav>
                    <ul>{currentUser ? <></> : <></>}</ul>
                </nav>
                <Routes>
                    {currentUser != null && currentUser.emailVerified ? (
                        <>
                            <Route path='/' element={<LandingPage />} />
                            <Route path='/dashboard' element={<Dashboard />} />
                            <Route path='/upload' element={<FileUpload />} />
                            <Route path='/old-runs' element={<OldRuns />} />
                        </>
                    ) : (
                        <>
                            <Route
                                path='/'
                                element={<Login manualSetCurrentUser={manualSetCurrentUser} />}
                            />
                            <Route path='/register' element={<Register />} />
                        </>
                    )}
                </Routes>
            </div>
        </Router>
    );
}

export default App;
