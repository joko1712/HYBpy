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

export const mainListItems = (navigate) => (
    <React.Fragment>
        <ListItemButton onClick={() => navigate("/Dashboard")}>
            <ListItemIcon>
                <HomeIcon />
            </ListItemIcon>
            <ListItemText primary='Home' />
        </ListItemButton>

        <ListItemButton onClick={() => navigate("/old-runs")}>
            <ListItemIcon>
                <InventoryIcon />
            </ListItemIcon>
            <ListItemText primary='Data' />
        </ListItemButton>

        <ListItemButton onClick={() => navigate("/upload")}>
            <ListItemIcon>
                <BackupIcon />
            </ListItemIcon>
            <ListItemText primary='Models' />
        </ListItemButton>

        <ListItemButton onClick={() => navigate("/")}>
            <ListItemIcon>
                <FlightLandIcon />
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
