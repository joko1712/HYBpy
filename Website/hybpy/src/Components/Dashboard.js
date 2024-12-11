import React, { useState, useEffect } from "react";
import { styled, createTheme, ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import MuiDrawer from "@mui/material/Drawer";
import Box from "@mui/material/Box";
import MuiAppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import MenuIcon from "@mui/icons-material/Menu";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import { mainListItems, secondaryListItems } from "./ListItems";
import { useNavigate, useLocation } from "react-router-dom";
import { auth } from "../firebase-config";
import {
    collection,
    query,
    where,
    getDocs,
    orderBy,
    limit,
} from "firebase/firestore";
import { db } from "../firebase-config";
import logo from "../Image/HYBpyINVIS_logo.png";
import ImageList from "@mui/material/ImageList";
import ImageListItem from "@mui/material/ImageListItem";
import Modal from "@mui/material/Modal";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Checkbox from "@mui/material/Checkbox";
import ListItemText from "@mui/material/ListItemText";
import Button from "@mui/material/Button";

const drawerWidth = 200;

const AppBar = styled(MuiAppBar, {
    shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
    zIndex: theme.zIndex.drawer + 1,
    transition: theme.transitions.create(["width", "margin"], {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.leavingScreen,
    }),
    ...(open && {
        marginLeft: drawerWidth,
        width: `calc(100% - ${drawerWidth}px)`,
        transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    }),
}));

const Drawer = styled(MuiDrawer, {
    shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
    "& .MuiDrawer-paper": {
        position: "relative",
        whiteSpace: "nowrap",
        width: drawerWidth,
        transition: theme.transitions.create("width", {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
        boxSizing: "border-box",
        ...(!open && {
            overflowX: "hidden",
            transition: theme.transitions.create("width", {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.leavingScreen,
            }),
            width: theme.spacing(7),
            [theme.breakpoints.up("sm")]: {
                width: theme.spacing(9),
            },
        }),
    },
}));

const defaultTheme = createTheme();

const style = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: "60%",
    height: "80%",
    bgcolor: "background.paper",
    border: "2px solid #000",
    boxShadow: 24,
    p: 4,
    overflow: "auto",
    alignItems: "center",
    justifyContent: "center",
};

export default function Dashboard() {
    const [runInProgress, setRunInProgress] = useState("");

    const [runs, setRuns] = useState([]);
    const [selectedPlot, setSelectedPlot] = useState(null);
    const [modalOpen, setModalOpen] = useState(false);
    const [mode, setMode] = useState("Error");
    const userId = auth.currentUser.uid;
    const [currentPlotIndex, setCurrentPlotIndex] = useState(0);
    const [selectedPlots, setSelectedPlots] = useState([]);
    const [showMetabolites, setShowMetabolites] = useState(true);

    const handlePlotSelectionChange = (event) => {
        const {
            target: { value },
        } = event;
        setSelectedPlots(typeof value === "string" ? value.split(",") : value);
    };

    const getPlotTitle = (url) => {
        const parts = url.split("/");
        const fileName = parts[parts.length - 1];
        if (url.includes("predicted_vs_observed")) {
            const species = fileName.split("_")[3].toUpperCase();
            return `PREDICTED VS OBSERVED - ${species}`;
        } else {
            const species = fileName.split("_")[1].toUpperCase();
            return `${species}`;
        }
    };

    const handleOpenModal = (index) => {
        const filteredPlots = getFilteredPlots();
        setCurrentPlotIndex(index);
        setSelectedPlot(filteredPlots[index]);
        setModalOpen(true);
    };

    const handleNextPlot = () => {
        const filteredPlots = getFilteredPlots();
        const nextIndex = (currentPlotIndex + 1) % filteredPlots.length;
        setCurrentPlotIndex(nextIndex);
        setSelectedPlot(filteredPlots[nextIndex]);
    };

    const handlePrevPlot = () => {
        const filteredPlots = getFilteredPlots();
        const prevIndex =
            (currentPlotIndex - 1 + filteredPlots.length) %
            filteredPlots.length;
        setCurrentPlotIndex(prevIndex);
        setSelectedPlot(filteredPlots[prevIndex]);
    };

    const getFilteredPlots = () => {
        const plots = runs[0]?.plots || [];
        if (showMetabolites) {
            return plots.filter((url) => url.includes("metabolite"));
        } else {
            return plots.filter((url) => url.includes("predicted_vs_observed"));
        }
    };

    const handleCloseModal = () => {
        setSelectedPlot(null);
        setModalOpen(false);
    };

    const checkRunStatus = async () => {
        try {
            const response = await fetch(`/run-status?user_id=${userId}`);
            const data = await response.json();



            setRunInProgress(data.status);
        } catch (error) {
            console.error("Error checking run status:", error);
        }
    };

    const toggleShowMetabolites = () => {
        setShowMetabolites(!showMetabolites);
    };

    useEffect(() => {
        const fetchLatestRun = async () => {
            const runsCollectionRef = collection(db, "users", userId, "runs");
            const q = query(
                runsCollectionRef,
                where("userId", "==", userId),
                orderBy("createdAt", "desc"),
                limit(1)
            );
            const querySnapshot = await getDocs(q);
            const latestRun = querySnapshot.docs.map((doc) => ({
                id: doc.id,
                ...doc.data(),
            }));
            setRuns(latestRun);
            if (latestRun.length > 0) {
                if (latestRun[0].mode === "1") {
                    setMode("Manual");
                } else {
                    setMode("Automatic");
                }

                if (latestRun[0].status === "simulation in progress") {
                    setRunInProgress("Simulation in progress...");
                }
                else if (latestRun[0].status === "training in progress") {
                    setRunInProgress("Training in progress...");
                }
                else if (latestRun[0].status === "error") {
                    setRunInProgress("Task failed");
                }
                else {
                    setRunInProgress("Task completed");
                }

                checkRunStatus();
            }
        };

        fetchLatestRun();
    }, [userId]);

    const navigate = useNavigate();
    const location = useLocation();
    const [open, setOpen] = React.useState(
        localStorage.getItem("drawerOpen") === "true"
    );
    const toggleDrawer = () => {
        setOpen(!open);
        localStorage.setItem("drawerOpen", !open);
    };
    const navigateToPage = (path) => {
        navigate(path);
    };

    return (
        <ThemeProvider theme={defaultTheme}>
            <Box sx={{ display: "flex" }}>
                <CssBaseline />
                <AppBar position='absolute' open={open}>
                    <Toolbar
                        sx={{
                            pr: "2px",
                        }}>
                        <IconButton
                            edge='start'
                            color='inherit'
                            aria-label='open drawer'
                            onClick={toggleDrawer}
                            sx={{
                                marginRight: "36px",
                                ...(open && { display: "none" }),
                            }}>
                            <MenuIcon />
                        </IconButton>
                        <Typography
                            component='h1'
                            variant='h6'
                            color='inherit'
                            noWrap
                            sx={{ flexGrow: 1 }}>
                            <IconButton
                                edge='start'
                                color='inherit'
                                size='small'
                                onClick={() => navigateToPage("/")}>
                                <img
                                    src={logo}
                                    alt='logo'
                                    width='200'
                                    height='75'
                                />
                            </IconButton>
                        </Typography>
                    </Toolbar>
                </AppBar>
                <Drawer variant='permanent' open={open}>
                    <Toolbar
                        sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "flex-end",
                            marginBottom: "4px",
                        }}>
                        <IconButton onClick={toggleDrawer}>
                            <ChevronLeftIcon />
                        </IconButton>
                    </Toolbar>
                    <Divider />
                    <List component='nav' disablePadding>
                        {mainListItems(navigate, location.pathname)}
                        <Divider sx={{ my: 1 }} />
                        {secondaryListItems(navigate)}
                    </List>
                </Drawer>
                <Box
                    component='main'
                    sx={{
                        backgroundColor: (theme) =>
                            theme.palette.mode === "light"
                                ? theme.palette.grey[100]
                                : theme.palette.grey[900],
                        flexGrow: 1,
                        height: "98vh",
                        overflow: "auto",
                        hideScrollbar: { scrollbarWidth: "none" },
                    }}>
                    <Toolbar />
                    <Container maxWidth='lg' sx={{ mt: 4, mb: 4, minHeight: "90%" }}>
                        <Grid container spacing={3}>
                            {/* Recent Run Details */}
                            <Grid item xs={12}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                    }}>
                                    {runs.length > 0 && runs[0].trained_weights ? (
                                        <>
                                            <Typography
                                                variant='h4'
                                                gutterBottom>
                                                Hybrid Model Details
                                            </Typography>
                                            <Typography variant='h6'>{`Title: ${runs[0].description}`}</Typography>
                                            <Typography>{`HMOD: ${runs[0].file1_name}`}</Typography>
                                            <Typography>{`CSV: ${runs[0].file2_name}`}</Typography>
                                            <Typography>{`Status: ${runInProgress}`}</Typography>
                                        </>
                                    )
                                        :
                                        runs.length > 0 ? (
                                            <>
                                                <Typography
                                                    variant='h4'
                                                    gutterBottom>
                                                    Hybrid Model Details
                                                </Typography>
                                                <Typography variant='h6'>{`Title: ${runs[0].description}`}</Typography>
                                                <Typography>{`HMOD: ${runs[0].file1_name}`}</Typography>
                                                <Typography>{`CSV: ${runs[0].file2_name}`}</Typography>

                                                <Typography>{`Mode: ${mode}`}</Typography>
                                                <Typography>{`Status: ${runInProgress}`}</Typography>
                                            </>
                                        ) : (
                                            <Typography>
                                                No recent run details
                                            </Typography>
                                        )}
                                </Paper>
                            </Grid>
                            {/* Recent Runs */}
                            <Grid item xs={12}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                    }}>
                                    <Button
                                        variant='contained'
                                        onClick={toggleShowMetabolites}
                                        sx={{ mb: 2 }}>
                                        {showMetabolites
                                            ? "Show Predicted vs Observed Plots"
                                            : "Show Metabolite Plots"}
                                    </Button>
                                    {getFilteredPlots().length > 0 ? (
                                        <ImageList
                                            cols={3}
                                            gap={8}
                                            sx={{ width: "100%" }}>
                                            {getFilteredPlots()
                                                .filter(
                                                    (url) =>
                                                        selectedPlots.length ===
                                                        0 ||
                                                        selectedPlots.includes(
                                                            url
                                                        )
                                                )
                                                .map((url, index) => {
                                                    const filteredPlots =
                                                        getFilteredPlots().filter(
                                                            (url) =>
                                                                selectedPlots.length ===
                                                                0 ||
                                                                selectedPlots.includes(
                                                                    url
                                                                )
                                                        );
                                                    const filteredIndex =
                                                        filteredPlots.indexOf(
                                                            url
                                                        );
                                                    return (
                                                        <ImageListItem
                                                            key={index}
                                                            onClick={() =>
                                                                handleOpenModal(
                                                                    filteredIndex
                                                                )
                                                            }>
                                                            <img
                                                                src={url}
                                                                alt={url}
                                                                loading='lazy'
                                                                style={{
                                                                    width: "100%",
                                                                    height: "auto",
                                                                    cursor: "pointer",
                                                                }}
                                                            />
                                                        </ImageListItem>
                                                    );
                                                })}
                                        </ImageList>
                                    ) : (
                                        <Typography>
                                            No plots available
                                        </Typography>
                                    )}
                                </Paper>
                            </Grid>
                            <Grid item xs={12}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                    }}>
                                    <Typography variant='h6'>
                                        Select Plots to Display
                                    </Typography>
                                    <Select
                                        multiple
                                        value={selectedPlots}
                                        onChange={handlePlotSelectionChange}
                                        renderValue={(selected) =>
                                            selected.length === 0
                                                ? "All Plots"
                                                : selected
                                                    .map(getPlotTitle)
                                                    .join(", ")
                                        }>
                                        {getFilteredPlots().map(
                                            (url, index) => {
                                                const plotTitle =
                                                    getPlotTitle(url);
                                                return (
                                                    <MenuItem
                                                        key={index}
                                                        value={url}>
                                                        <Checkbox
                                                            checked={
                                                                selectedPlots.indexOf(
                                                                    url
                                                                ) > -1
                                                            }
                                                        />
                                                        <ListItemText
                                                            primary={plotTitle}
                                                        />
                                                    </MenuItem>
                                                );
                                            }
                                        )}
                                    </Select>
                                </Paper>
                            </Grid>
                        </Grid>
                        <Modal
                            open={modalOpen}
                            onClose={handleCloseModal}
                            aria-labelledby='modal-modal-title'
                            aria-describedby='modal-modal-description'>
                            <Box
                                sx={{
                                    ...style,
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    position: "relative",
                                }}>
                                {selectedPlot && (
                                    <>
                                        <IconButton
                                            onClick={handlePrevPlot}
                                            sx={{
                                                position: "absolute",
                                                top: "50%",
                                                left: "16px",
                                                transform: "translateY(-50%)",
                                                zIndex: 1,
                                            }}>
                                            <ArrowBackIcon />
                                        </IconButton>
                                        <img
                                            src={selectedPlot}
                                            alt='Selected Plot'
                                            style={{
                                                maxWidth: "100%",
                                                maxHeight: "90%",
                                                display: "block",
                                                margin: "0 auto",
                                            }}
                                        />
                                        <IconButton
                                            onClick={handleNextPlot}
                                            sx={{
                                                position: "absolute",
                                                top: "50%",
                                                right: "16px",
                                                transform: "translateY(-50%)",
                                                zIndex: 1,
                                            }}>
                                            <ArrowForwardIcon />
                                        </IconButton>
                                    </>
                                )}
                            </Box>
                        </Modal>
                    </Container>
                    <Box
                        component='main'
                        sx={{
                            display: "flex",
                            flexDirection: "column",
                            flexGrow: 1,
                            backgroundColor: (theme) => theme.palette.grey[100],
                        }}>
                        <Toolbar />
                        <Container maxWidth='lg' ></Container>

                        <Box
                            component='footer'
                            sx={{
                                p: 2,
                                backgroundColor: "#f1f1f1",
                                position: "fixed",
                                bottom: 0,
                                left: open ? `${drawerWidth}px` : "56px", // Adjust based on drawer state
                                width: open
                                    ? `calc(100% - ${drawerWidth}px)`
                                    : "calc(100% - 56px)",
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                transition: "width 0.3s ease, left 0.3s ease", // Smooth transition when toggling drawer
                            }}>
                            <Typography
                                variant='body2'
                                align='center'
                                sx={{ flexGrow: 1 }}>
                                &copy; {new Date().getFullYear()} Faculdade de
                                Ciências e Tecnologia Universidade NOVA de
                                Lisboa. All rights reserved.
                            </Typography>
                            <img
                                src='https://www.fct.unl.pt/sites/default/files/images/logo_nova_fct_pt_v.png'
                                width='75'
                                alt='FCT Logo'
                                style={{ marginLeft: "auto" }}
                            />
                        </Box>
                    </Box>
                </Box>
            </Box>
        </ThemeProvider>
    );
}
