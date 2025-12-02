import * as React from "react";
import { styled, alpha } from "@mui/material/styles";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import BackupIcon from "@mui/icons-material/Backup";
import LogoutIcon from "@mui/icons-material/Logout";
import FaceIcon from "@mui/icons-material/Face";
import InventoryIcon from "@mui/icons-material/Inventory";
import HomeIcon from "@mui/icons-material/Home";
import BatchPredictionIcon from "@mui/icons-material/BatchPrediction";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import SupportIcon from "@mui/icons-material/Support";
import { auth } from "../firebase-config";

const CustomListItemButton = styled(ListItemButton)(({ theme }) => ({
    paddingTop: theme.spacing(1),
    paddingBottom: theme.spacing(1),
    "&.Mui-selected": {
        backgroundColor: alpha(theme.palette.primary.main, 0.75),
        color: theme.palette.common.white,
        "& .MuiListItemIcon-root": {
            color: theme.palette.common.white,
        },
    },
    "&.Mui-selected:hover": {
        backgroundColor: theme.palette.primary.main,
    },
}));

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
        <CustomListItemButton
            onClick={() => navigate("/")}
            selected={locationPath === "/"}>
            <ListItemIcon>
                <HomeIcon />
            </ListItemIcon>
            <ListItemText primary='Home' />
        </CustomListItemButton>

        <CustomListItemButton
            onClick={() => navigate("/upload")}
            selected={locationPath === "/upload"}>
            <ListItemIcon>
                <BackupIcon />
            </ListItemIcon>
            <ListItemText primary='New Project' />
        </CustomListItemButton>

        <CustomListItemButton
            onClick={() => navigate("/results")}
            selected={locationPath === "/results"}>
            <ListItemIcon>
                <QueryStatsIcon />
            </ListItemIcon>
            <ListItemText primary='Results' />
        </CustomListItemButton>

        <CustomListItemButton
            onClick={() => navigate("/simulations")}
            selected={locationPath === "/simulations"}>
            <ListItemIcon>
                <BatchPredictionIcon />
            </ListItemIcon>
            <ListItemText primary='Simulation' />
        </CustomListItemButton>

        <CustomListItemButton
            onClick={() => navigate("/historical")}
            selected={locationPath === "/historical"}>
            <ListItemIcon>
                <InventoryIcon />
            </ListItemIcon>
            <ListItemText primary='Historical' />
        </CustomListItemButton>

        <CustomListItemButton
            onClick={() => navigate("/help")}
            selected={locationPath === "/help"}>
            <ListItemIcon>
                <SupportIcon />
            </ListItemIcon>
            <ListItemText primary='Help' />
        </CustomListItemButton>
    </React.Fragment>
);

export const secondaryListItems = (navigate) => (
    <React.Fragment>
        <CustomListItemButton>
            <ListItemIcon>
                <FaceIcon />
            </ListItemIcon>
            <ListItemText
                primary={auth.currentUser.displayName || auth.currentUser.email}
            />
        </CustomListItemButton>
        <CustomListItemButton onClick={() => logout(navigate)}>
            <ListItemIcon>
                <LogoutIcon />
            </ListItemIcon>
            <ListItemText primary='LogOut' />
        </CustomListItemButton>
    </React.Fragment>
);
