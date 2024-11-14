import * as React from "react";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import BackupIcon from "@mui/icons-material/Backup";
import LogoutIcon from "@mui/icons-material/Logout";
import FaceIcon from "@mui/icons-material/Face";
import InventoryIcon from "@mui/icons-material/Inventory";
import FlightLandIcon from "@mui/icons-material/FlightLand";
import HomeIcon from "@mui/icons-material/Home";
import BatchPredictionIcon from "@mui/icons-material/BatchPrediction";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import SupportIcon from "@mui/icons-material/Support";
import { auth } from "../firebase-config";

const logout = (navigate) => {
    auth.signOut()
        .then(() => {
            navigate("/");
        })
        .catch((error) => {
            console.error("Logout error:", error);
        });
};

export const mainListItems = (navigate, locationPath) => (
    <React.Fragment>
        <ListItemButton
            onClick={() => navigate("/")}
            selected={locationPath === "/"}
        >
            <ListItemIcon>
                <HomeIcon />
            </ListItemIcon>
            <ListItemText primary='Home' />
        </ListItemButton>

        <ListItemButton
            onClick={() => navigate("/upload")}
            selected={locationPath === "/upload"}
        >
            <ListItemIcon>
                <BackupIcon />
            </ListItemIcon>
            <ListItemText primary='New Project' />
        </ListItemButton>

        <ListItemButton
            onClick={() => navigate("/simulations")}
            selected={locationPath === "/simulations"}
        >
            <ListItemIcon>
                <BatchPredictionIcon />
            </ListItemIcon>
            <ListItemText primary='Simulation' />
        </ListItemButton>

        <ListItemButton
            onClick={() => navigate("/results")}
            selected={locationPath === "/results"}
        >
            <ListItemIcon>
                <QueryStatsIcon />
            </ListItemIcon>
            <ListItemText primary='Results' />
        </ListItemButton>

        <ListItemButton
            onClick={() => navigate("/historical")}
            selected={locationPath === "/historical"}
        >
            <ListItemIcon>
                <InventoryIcon />
            </ListItemIcon>
            <ListItemText primary='Historical' />
        </ListItemButton>

        <ListItemButton
            onClick={() => navigate("/help")}
            selected={locationPath === "/help"}
        >
            <ListItemIcon>
                <SupportIcon />
            </ListItemIcon>
            <ListItemText primary='Help' />
        </ListItemButton>
    </React.Fragment>
);

export const secondaryListItems = (navigate) => (
    <React.Fragment>
        <ListItemButton>
            <ListItemIcon>
                <FaceIcon />
            </ListItemIcon>
            <ListItemText primary={auth.currentUser.displayName || auth.currentUser.email} />
        </ListItemButton>
        <ListItemButton onClick={() => logout(navigate)}>
            <ListItemIcon>
                <LogoutIcon />
            </ListItemIcon>
            <ListItemText primary='LogOut' />
        </ListItemButton>
    </React.Fragment>
);
