import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

import RegisterLogin from "./Components/RegisterLogin";
import FileUpload from "./Components/FileUpload";
import OldRuns from "./Components/OldRuns";
import Register from "./Components/Register";
import Login from "./Components/Login";

import { auth } from "./firebase-config";

function App() {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(false);
    const [hasShownRegisterLogin, setHasShownRegisterLogin] = useState(false); // New state variable

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

    if (loading) {
        return <div>Loading...</div>;
    }

    return (
        <Router>
            <div>
                <nav>
                    <ul>
                        {currentUser ? (
                            <>
                                <li>
                                    <Link to='/upload'>Upload</Link>
                                </li>
                                <li>
                                    <Link to='/old-runs'>Old Runs</Link>
                                </li>
                                <li>
                                    <button onClick={() => auth.signOut()}>Logout</button>
                                </li>
                            </>
                        ) : (
                            <></>
                        )}
                    </ul>
                </nav>
                <Routes>
                    {currentUser ? (
                        <>
                            <Route path='/upload' element={<FileUpload />} />
                            <Route path='/old-runs' element={<OldRuns />} />
                        </>
                    ) : (
                        <>
                            {!hasShownRegisterLogin && (
                                <Route path='/' element={<RegisterLogin />} />
                            )}
                            <Route path='/login' element={<Login />} />
                            <Route path='/register' element={<Register />} />
                        </>
                    )}
                </Routes>
            </div>
        </Router>
    );
}

export default App;
