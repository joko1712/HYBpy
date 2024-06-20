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
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import { mainListItems, secondaryListItems } from "./ListItems";
import { useNavigate } from "react-router-dom";
import { auth } from "../firebase-config";
import { collection, query, where, getDocs, orderBy, limit } from "firebase/firestore";
import { db } from "../firebase-config";
import logo from "../Image/HYBpyINVIS_logo.png";
import ImageList from "@mui/material/ImageList";
import ImageListItem from "@mui/material/ImageListItem";
import Modal from "@mui/material/Modal";

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

const Drawer = styled(MuiDrawer, { shouldForwardProp: (prop) => prop !== "open" })(
    ({ theme, open }) => ({
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
    })
);

const defaultTheme = createTheme();

const style = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: "80%",
    height: "80%",
    bgcolor: "background.paper",
    border: "2px solid #000",
    boxShadow: 24,
    p: 4,
    overflow: "auto",
};
export default function Dashboard() {
    const navigate = useNavigate();

    const navigateToUpload = () => {
        navigate("/Dashboard");
    };
    const [open, setOpen] = useState(true);
    const [runInProgress, setRunInProgress] = useState("");
    const toggleDrawer = () => {
        setOpen(!open);
    };

    const [runs, setRuns] = useState([]);
    const [selectedPlot, setSelectedPlot] = useState(null);
    const [modalOpen, setModalOpen] = useState(false);
    const [mode, setMode] = useState("Error");
    const userId = auth.currentUser.uid;

    const handleOpenModal = (url) => {
        setSelectedPlot(url);
        setModalOpen(true);
    };

    const handleCloseModal = () => {
        setSelectedPlot(null);
        setModalOpen(false);
    };

    const checkRunStatus = async () => {
        try {
            const response = await fetch(`/run-status?user_id=${userId}`);
            const data = await response.json();
            setRunInProgress(data.status === "in_progress");
        } catch (error) {
            console.error("Error checking run status:", error);
        }
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
            if (latestRun[0].mode === 1) {
                setMode("Manual");
            } else {
                setMode("Automatic");
            }

            if (latestRun[0].status === "in_progress") {
                setRunInProgress("Trainning in progress...");
            } else {
                setRunInProgress("Trainning completed");
            }

            checkRunStatus();
        };

        fetchLatestRun();
    }, [userId]);

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
                                onClick={() => navigateToUpload()}>
                                <img src={logo} alt='logo' width='200' height='75' />
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
                        }}>
                        <IconButton onClick={toggleDrawer}>
                            <ChevronLeftIcon />
                        </IconButton>
                    </Toolbar>
                    <Divider />
                    <List component='nav'>
                        {mainListItems(navigate)}
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
                    }}>
                    <Toolbar />
                    <Container maxWidth='lg' sx={{ mt: 4, mb: 4 }}>
                        <Grid container spacing={3}>
                            {/* Recent Run Details */}
                            <Grid item xs={12}>
                                <Paper sx={{ p: 2, display: "flex", flexDirection: "column" }}>
                                    {runs.length > 0 ? (
                                        <>
                                            <Typography variant='h4' gutterBottom>
                                                Hybrid Model Details
                                            </Typography>
                                            <Typography variant='h6'>{`Title: ${runs[0].description}`}</Typography>
                                            <Typography>{`HMOD: ${runs[0].file1_name}`}</Typography>
                                            <Typography>{`CSV: ${runs[0].file2_name}`}</Typography>
                                            <Typography>{`Mode: ${mode}`}</Typography>
                                            <Typography>{`Status: ${runInProgress}`}</Typography>
                                        </>
                                    ) : (
                                        <Typography>No recent run details</Typography>
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
                                    {runs.length > 0 && runs[0].plots ? (
                                        <ImageList cols={3} gap={8} sx={{ width: "100%" }}>
                                            {runs[0].plots.map((url, index) => (
                                                <ImageListItem
                                                    key={index}
                                                    onClick={() => handleOpenModal(url)}>
                                                    <img
                                                        src={url}
                                                        alt={`Plot ${index}`}
                                                        loading='lazy'
                                                        style={{
                                                            width: "100%",
                                                            height: "auto",
                                                            cursor: "pointer",
                                                        }}
                                                    />
                                                </ImageListItem>
                                            ))}
                                        </ImageList>
                                    ) : (
                                        <Typography>No plots available</Typography>
                                    )}
                                </Paper>
                            </Grid>
                        </Grid>
                        <Modal
                            open={modalOpen}
                            onClose={handleCloseModal}
                            aria-labelledby='modal-modal-title'
                            aria-describedby='modal-modal-description'>
                            <Box sx={style}>
                                {selectedPlot && (
                                    <img
                                        src={selectedPlot}
                                        alt='Selected Plot'
                                        style={{ width: "auto", height: "100%" }}
                                    />
                                )}
                            </Box>
                        </Modal>
                    </Container>
                </Box>
            </Box>
        </ThemeProvider>
    );
}
