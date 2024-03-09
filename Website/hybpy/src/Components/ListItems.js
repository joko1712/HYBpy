import * as React from "react";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import BackupIcon from "@mui/icons-material/Backup";
import LogoutIcon from "@mui/icons-material/Logout";
import FaceIcon from "@mui/icons-material/Face";
import InventoryIcon from "@mui/icons-material/Inventory";
import FlightLandIcon from "@mui/icons-material/FlightLand";
import { auth } from "../firebase-config";

const logout = (navigate) => {
    auth.signOut();
    navigate("/");
};
export const mainListItems = (navigate) => (
    <React.Fragment>
        <ListItemButton onClick={() => navigate("/landingPage")}>
            <ListItemIcon>
                <FlightLandIcon />
            </ListItemIcon>
            <ListItemText primary='Landing Page' />
        </ListItemButton>
        <ListItemButton>
            <ListItemIcon>
                <FaceIcon />
            </ListItemIcon>
            <ListItemText primary={auth.currentUser.displayName} />
        </ListItemButton>
        <ListItemButton onClick={() => navigate("/upload")}>
            <ListItemIcon>
                <BackupIcon />
            </ListItemIcon>
            <ListItemText primary='Create Run' />
        </ListItemButton>
        <ListItemButton onClick={() => navigate("/old-runs")}>
            <ListItemIcon>
                <InventoryIcon />
            </ListItemIcon>
            <ListItemText primary='Old Runs' />
        </ListItemButton>
    </React.Fragment>
);

export const secondaryListItems = (navigate) => (
    <React.Fragment>
        <ListItemButton onClick={() => logout(navigate)}>
            <ListItemIcon>
                <LogoutIcon />
            </ListItemIcon>
            <ListItemText primary='LogOut' />
        </ListItemButton>
    </React.Fragment>
);
