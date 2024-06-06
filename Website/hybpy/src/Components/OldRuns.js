import * as React from "react";
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
import { collection, query, where, getDocs, orderBy } from "firebase/firestore";
import { db } from "../firebase-config";
import { useEffect } from "react";
import ListItemText from "@mui/material/ListItemText";
import ListIcon from "@mui/icons-material/List";
import { blue } from "@mui/material/colors";
import { ListItemButton } from "@mui/material";
import Modal from "@mui/material/Modal";
import logo from "../Image/HYBpyINVIS_logo.png";
import ImageList from "@mui/material/ImageList";
import ImageListItem from "@mui/material/ImageListItem";

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

function DisplayJson({ data, level = 0 }) {
    if (data === null || data === undefined) {
        return <span>No data available</span>;
    }
    if (typeof data !== "object") {
        return <span>{data.toString()}</span>;
    }

    return (
        <ul style={{ marginLeft: level * 20, paddingTop: -20 }}>
            {Object.entries(data).map(([key, value]) => (
                <li key={key}>
                    <strong>{key}:</strong>{" "}
                    {typeof value === "object" ? (
                        <DisplayJson data={value} level={level + 1} />
                    ) : (
                        value.toString()
                    )}
                </li>
            ))}
        </ul>
    );
}

const defaultTheme = createTheme();

export default function OldRuns() {
    const navigate = useNavigate();

    const navigateToUpload = () => {
        navigate("/Dashboard");
    };
    const [open, setOpen] = React.useState(true);
    const toggleDrawer = () => {
        setOpen(!open);
    };

    const [runs, setRuns] = React.useState([]);
    const userId = auth.currentUser.uid;

    const [openModal, setOpenModal] = React.useState(false);
    const [selectedRun, setSelectedRun] = React.useState(null);
    const [selectedPlot, setSelectedPlot] = React.useState(null);

    const handleOpen = (run) => {
        setSelectedRun(run);
        setOpenModal(true);
    };

    const handleClose = () => {
        setOpenModal(false);
        setSelectedPlot(null);
    };

    const handlePlotClick = (url) => {
        setSelectedPlot(url);
    };

    useEffect(() => {
        const fetchLatestRun = async () => {
            const runsCollectionRef = collection(db, "users", userId, "runs");
            const q = query(
                runsCollectionRef,
                where("userId", "==", userId),
                orderBy("createdAt", "desc")
            );
            const querySnapshot = await getDocs(q);
            const latestRun = querySnapshot.docs.map((doc) => ({
                id: doc.id,
                ...doc.data(),
            }));
            setRuns(latestRun);
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
                        hideScrollbar: { scrollbarWidth: "none" },
                    }}>
                    <Toolbar />
                    <Container maxWidth='lg' sx={{ mt: 1, mb: 4 }}>
                        <h2>Old Runs</h2>
                        <Grid container spacing={3}>
                            <Grid item xs={20}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        height: 350,
                                    }}>
                                    <List>
                                        {runs.map((run) => (
                                            <ListItemButton
                                                key={run.id}
                                                onClick={() => handleOpen(run)}>
                                                <ListItemText
                                                    primary={`Run ID: ${run.id}`}
                                                    secondary={`Hmod: ${run.file1_name} - CSV: ${run.file2_name} -Mode: ${run.mode}`}
                                                />
                                                <p>View Run: </p>
                                                <IconButton
                                                    sx={{ color: blue[500] }}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleOpen(run);
                                                    }}>
                                                    <ListIcon fontSize='large' />
                                                </IconButton>
                                            </ListItemButton>
                                        ))}
                                    </List>
                                    <Modal
                                        open={openModal}
                                        onClose={handleClose}
                                        aria-labelledby='modal-modal-title'
                                        aria-describedby='modal-modal-description'>
                                        <Box sx={{ ...style, width: "80%", height: "80%" }}>
                                            {selectedRun && (
                                                <>
                                                    <Typography
                                                        id='modal-modal-title'
                                                        variant='h6'
                                                        component='h2'>
                                                        Run Details
                                                    </Typography>
                                                    <Typography
                                                        id='modal-modal-description'
                                                        sx={{ mt: 3 }}>
                                                        Plots:
                                                        <div>
                                                            {selectedRun.plots &&
                                                                selectedRun.plots.map(
                                                                    (plotUrl, index) => (
                                                                        <img
                                                                            key={index}
                                                                            src={plotUrl}
                                                                            alt={`Plot ${index}`}
                                                                            style={{
                                                                                width: "30%",
                                                                                marginBottom: 10,
                                                                                cursor: "pointer",
                                                                            }}
                                                                            onClick={() =>
                                                                                handlePlotClick(
                                                                                    plotUrl
                                                                                )
                                                                            }
                                                                        />
                                                                    )
                                                                )}
                                                        </div>
                                                        Run ID: {selectedRun.id}
                                                        <br />
                                                        Hmod: {selectedRun.file1_name}
                                                        <br />
                                                        CSV: {selectedRun.file2_name}
                                                        <br />
                                                        Mode: {selectedRun.mode}
                                                        <br />
                                                        Description: {selectedRun.description}
                                                        <br />
                                                        Response:{" "}
                                                        {selectedRun.response_data.message}
                                                        <br />
                                                        Projhyb:
                                                        <DisplayJson
                                                            data={selectedRun.response_data.projhyb}
                                                        />
                                                        <br />
                                                        <DisplayJson
                                                            data={
                                                                selectedRun.response_data.trainData
                                                            }
                                                        />
                                                    </Typography>
                                                </>
                                            )}
                                        </Box>
                                    </Modal>
                                    <Modal
                                        open={!!selectedPlot}
                                        onClose={() => setSelectedPlot(null)}
                                        aria-labelledby='modal-plot-title'
                                        aria-describedby='modal-plot-description'>
                                        <Box
                                            sx={{
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
                                            }}>
                                            <img
                                                src={selectedPlot}
                                                alt='Selected Plot'
                                                style={{ width: "100%" }}
                                            />
                                        </Box>
                                    </Modal>
                                </Paper>
                            </Grid>
                        </Grid>
                    </Container>
                </Box>
            </Box>
        </ThemeProvider>
    );
}
