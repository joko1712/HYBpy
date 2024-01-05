import React from "react";
import { useNavigate } from "react-router-dom";

function RegisterLogin() {
    const navigate = useNavigate();

    const handleNavigate = (path) => {
        navigate(path);
    };

    return (
        <div>
            Choose an option Below:
            <br />
            <button onClick={() => handleNavigate("/register")}>Register</button>
            <button onClick={() => handleNavigate("/login")}>Login</button>
        </div>
    );
}

export default RegisterLogin;
