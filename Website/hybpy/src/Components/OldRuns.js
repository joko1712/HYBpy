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
import { useNavigate, useLocation } from "react-router-dom";
import { auth } from "../firebase-config";
import { collection, query, where, getDocs, orderBy } from "firebase/firestore";
import { db } from "../firebase-config";
import { useEffect } from "react";
import ListItemText from "@mui/material/ListItemText";
import CloseIcon from "@mui/icons-material/Close";
import ListItemIcon from "@mui/material/ListItemIcon";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { red } from "@mui/material/colors";
import Modal from "@mui/material/Modal";
import logo from "../Image/HYBpyINVIS_logo.png";
import DeleteIcon from "@mui/icons-material/Delete";
import {
    ListItemButton,
    TextField,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    Button,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
} from "@mui/material";
import BackupIcon from "@mui/icons-material/Backup";
import BatchPredictionIcon from "@mui/icons-material/BatchPrediction";
import { handleContactUsClick } from "./ContactUs";
import EmailIcon from '@mui/icons-material/Email';



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

const DisplayJson = ({ data }) => {
    console.log(data);

    const isArrayOfObjects = data.length > 0 && typeof data[0] === "object";

    if (!isArrayOfObjects) {
        return (
            <div style={{ whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
                {JSON.stringify(data)}
            </div>
        );
    }

    return (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
                <tr>
                    {Object.keys(data[0]).map((key) => (
                        <th
                            key={key}
                            style={{
                                border: "1px solid black",
                                padding: "5px",
                            }}>
                            {key}
                        </th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {data.map((item, index) => (
                    <tr key={index}>
                        {Object.values(item).map((value, i) => (
                            <td
                                key={i}
                                style={{
                                    border: "1px solid black",
                                    padding: "5px",
                                }}>
                                {value}
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
};

const defaultTheme = createTheme();

export default function OldRuns() {
    const [runs, setRuns] = React.useState([]);
    const [searchQuery, setSearchQuery] = React.useState("");
    const userId = auth.currentUser.uid;
    const [mode, setMode] = React.useState("Error");
    const [date, setDate] = React.useState("Error");

    const [selectedPlotIndex, setSelectedPlotIndex] = React.useState(null);
    const [openModal, setOpenModal] = React.useState(false);
    const [selectedRun, setSelectedRun] = React.useState(null);
    const [selectedPlot, setSelectedPlot] = React.useState(null);
    const [deleteDialogOpen, setDeleteDialogOpen] = React.useState(false);
    const [runToDelete, setRunToDelete] = React.useState(null);
    const [fileUrls, setFileUrls] = React.useState({
        file1_url: "",
        file2_url: "",
        new_hmod_url: "",
    });

    const [isloading, setIsLoading] = React.useState(false);

    const handleOpen = async (run) => {
        setSelectedRun(run);
        console.log("Selected Run:", run);
        try {
            const urls = await fetchFileUrls(userId, run.id);
            setFileUrls(urls);
        } catch (error) {
            console.error("Error fetching file URLs:", error);
        }
        setOpenModal(true);
    };

    const handleClose = () => {
        setOpenModal(false);
        setSelectedPlot(null);
    };

    const handleDeleteDialogOpen = (run) => {
        setRunToDelete(run);
        setDeleteDialogOpen(true);
    };

    const handleDeleteDialogClose = () => {
        setRunToDelete(null);
        setDeleteDialogOpen(false);
    };

    const handlePlotClick = (index) => {
        setSelectedPlotIndex(index);
    };

    const handlePrevPlot = () => {
        setSelectedPlotIndex((prevIndex) => {
            const newIndex =
                prevIndex > 0 ? prevIndex - 1 : selectedRun.plots.length - 1;
            return newIndex;
        });
    };

    const handleNextPlot = () => {
        setSelectedPlotIndex((prevIndex) => {
            const newIndex = (prevIndex + 1) % selectedRun.plots.length;
            return newIndex;
        });
    };

    const fetchFileUrls = async (userId, runId) => {
        const response = await fetch(
            `https://api.hybpy.com/get-file-urls?user_id=${userId}&run_id=${runId}`
        );
        const data = response.json();
        if (response.ok) {
            return data;
        } else {
            throw new Error(data.error);
        }
    };

    const deleteRun = async () => {
        setIsLoading(true);
        try {
            const file1Url = runToDelete.file1;
            const folderPath = file1Url.split("/").slice(4, -1).join("/");

            const response = await fetch(
                "https://api.hybpy.com/delete-run",
                {
                    method: "DELETE",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        user_id: userId,
                        run_id: runToDelete.id,
                        folder_path: folderPath,
                    }),
                }
            );
            const result = await response.json();
            console.log("Delete response:", result);
            if (response.ok) {
                setRuns((prevRuns) =>
                    prevRuns.filter((run) => run.id !== runToDelete.id)
                );
                handleDeleteDialogClose();
            } else {
                console.error("Error deleting run:", result.error);
            }
        } catch (error) {
            console.error("Error deleting run:", error);
        }
        setIsLoading(false);
    };

    const getDisplayValue = (key, value) => {
        const mappings = {
            Jacobian: { 1: "On", 0: "Off" },
            Hessian: { 1: "On", 0: "Off" },
            Bootstrap: { 1: "On", 0: "Off" },
            Method: {
                1: "TRF",
                2: "Trust-Constr",
                3: "Simulated Annealing",
                4: "ADAM",
                5: "ADAM + ODEint",
            },
            Mode: { 1: "Indirect", 2: "Direct", 3: "Indirect" },
            Layer: { 1: "Tanh", 2: "ReLU", 3: "LSTM" },
        };

        return mappings[key] && mappings[key][value]
            ? mappings[key][value]
            : value;
    };


    const filteredRuns = runs.filter((run) => {
        return (
            searchQuery === "" ||
            run.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
            run.file1_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            run.file2_name.toLowerCase().includes(searchQuery.toLowerCase())
        );
    });

    const Modesetter = (mode) => {
        if (mode === 1) {
            return "Manual";
        } else if (mode === 2) {
            return "Automatic";
        } else {
            return "Error";
        }
    };

    const navigate = useNavigate();
    const location = useLocation();
    const selectedRunIdFromState = location.state?.selectedRunId;

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

    const machineLearningOptions = [
        { key: "HiddenNodes", label: "Hidden Nodes" },
        { key: "Layer", label: "Layer" },
        { key: "Tau", label: "TAU" },
        { key: "Mode", label: "Mode" },
        { key: "Method", label: "Method" },
        { key: "Niter", label: "Niter" },
        { key: "Nstep", label: "Nstep" },
        { key: "Jacobian", label: "Jacobian" },
        { key: "Hessian", label: "Hessian" },
        { key: "Bootstrap", label: "Bootstrap" },
    ];

    const calculateDuration = (startTime, endTime) => {
        const start = startTime.toDate();
        const end = endTime.toDate();
        const durationMillis = end - start;

        if (durationMillis < 0) {
            return "Invalid duration";
        }

        const durationSeconds = Math.floor(durationMillis / 1000);

        const days = Math.floor(durationSeconds / 86400);
        const hours = Math.floor((durationSeconds % 86400) / 3600);
        const minutes = Math.floor((durationSeconds % 3600) / 60);
        const seconds = durationSeconds % 60;

        let durationString = "";
        if (days > 0) {
            durationString += `${days}d `;
        }
        if (hours > 0 || days > 0) {
            durationString += `${hours}h `;
        }
        if (minutes > 0 || hours > 0 || days > 0) {
            durationString += `${minutes}m `;
        }
        durationString += `${seconds}s`;

        return durationString;
    };

    useEffect(() => {
        const fetchLatestRun = async () => {
            const runsCollectionRef = collection(db, "users", userId, "runs");
            const q = query(
                runsCollectionRef,
                where("status", "==", "completed"),
                where("userId", "==", userId),
                orderBy("createdAt", "desc")
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

                if (selectedRunIdFromState) {
                    const matchedRun = latestRun.find(
                        (run) => run.id === selectedRunIdFromState
                    );
                    if (matchedRun) {
                        handleOpen(matchedRun);
                        navigate(location.pathname, { replace: true });
                    }
                }
            }
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
                    <Container
                        maxWidth='lg'
                        sx={{ mt: 4, mb: 4, minHeight: "90%" }}>
                        <h1>
                            <strong>List of Historical Projects</strong>
                        </h1>
                        <Box sx={{ mb: 2 }}>
                            <TextField
                                label='Search Project'
                                variant='outlined'
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                fullWidth
                                sx={{ mb: 2 }}
                            />
                        </Box>
                        <Grid container spacing={3}>
                            <Grid item xs={12}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                    }}>
                                    <List>
                                        {filteredRuns.map((run) => (
                                            <ListItemButton
                                                key={run.id}
                                                onClick={() => handleOpen(run)}>
                                                {run.trained_weights ? (
                                                    <BatchPredictionIcon
                                                        sx={{ mr: 2 }}
                                                        titleAccess='Trained weights available'
                                                    />
                                                ) : (
                                                    <BackupIcon
                                                        sx={{ mr: 2 }}
                                                        titleAccess='Trained weights not available'
                                                    />
                                                )}
                                                <ListItemText
                                                    primary={`Title: ${run.description}`}
                                                    secondary={`HMOD:${
                                                        run.file1_name
                                                    } - CSV:${
                                                        run.file2_name
                                                    } - Mode:${mode} - CreatedAt:${run.createdAt
                                                        .toDate()
                                                        .toLocaleString()} -  FinishedAt: ${
                                                        run.finishedAt
                                                            ? run.finishedAt
                                                                  .toDate()
                                                                  .toLocaleString()
                                                            : "Not finished"
                                                    }`}
                                                />
                                                <IconButton
                                                    sx={{ color: red[500] }}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleDeleteDialogOpen(
                                                            run
                                                        );
                                                    }}>
                                                    <DeleteIcon fontSize='large' />
                                                </IconButton>
                                            </ListItemButton>
                                        ))}
                                    </List>
                                    <Modal
                                        open={openModal}
                                        onClose={handleClose}
                                        aria-labelledby='modal-modal-title'
                                        aria-describedby='modal-modal-description'>
                                        <Box
                                            sx={{
                                                ...style,
                                                width: { xs: "95%", md: "80%" },
                                                height: {
                                                    xs: "auto",
                                                    md: "90%",
                                                },
                                                overflowY: "auto",
                                            }}>
                                            {selectedRun && (
                                                <div
                                                    style={{
                                                        display: "flex",
                                                        flexDirection: "column",
                                                    }}>
                                                    <Typography
                                                        id='modal-modal-title'
                                                        variant='h4'
                                                        component='h2'>
                                                        <strong>
                                                            Title:{" "}
                                                            {
                                                                selectedRun.description
                                                            }
                                                        </strong>
                                                    </Typography>

                                                    <Typography
                                                        id='modal-modal-title'
                                                        variant='h4'
                                                        component='h2'
                                                        style={{
                                                            display: "flex",
                                                            alignSelf: "center",
                                                        }}>
                                                        Model Evaluation
                                                    </Typography>

                                                    {/* Display Plots */}
                                                    <div>
                                                        {selectedRun.plots &&
                                                            selectedRun.plots.map(
                                                                (
                                                                    plotUrl,
                                                                    index
                                                                ) => (
                                                                    <img
                                                                        key={
                                                                            index
                                                                        }
                                                                        src={
                                                                            plotUrl
                                                                        }
                                                                        alt={`Plot ${index}`}
                                                                        style={{
                                                                            width: "30%",
                                                                            marginBottom: 10,
                                                                            cursor: "pointer",
                                                                        }}
                                                                        onClick={() =>
                                                                            handlePlotClick(
                                                                                index
                                                                            )
                                                                        }
                                                                    />
                                                                )
                                                            )}
                                                    </div>

                                                    {selectedRun.response_data ? (
                                                        <>
                                                            <TableContainer
                                                                component={
                                                                    Paper
                                                                }
                                                                sx={{
                                                                    marginTop: 3,
                                                                }}>
                                                                <Table aria-label='data table'>
                                                                    <TableBody>
                                                                        <TableRow>
                                                                            <TableCell
                                                                                colSpan={
                                                                                    2
                                                                                }
                                                                                sx={{
                                                                                    backgroundColor:
                                                                                        "#f5f5f5",
                                                                                }}>
                                                                                <Typography variant='h6'>
                                                                                    Metrics
                                                                                </Typography>
                                                                            </TableCell>
                                                                        </TableRow>

                                                                        
                                                                        {selectedRun.response_data.metrics ? (
                                                                            ["r2_train", "r2_test", "mse_train", "mse_test"].map((key) => {
                                                                                const value = selectedRun.response_data.metrics[key];
                                                                                if (value === undefined) return null;
                                                                                return (
                                                                                    <TableRow key={key}>
                                                                                        <TableCell component='th' scope='row'>
                                                                                            {key}
                                                                                        </TableCell>
                                                                                        <TableCell>
                                                                                            {typeof value === "number" ? value.toFixed(5) : value}
                                                                                        </TableCell>
                                                                                    </TableRow>
                                                                                );
                                                                            })
                                                                        ) : (
                                                                            <TableRow>
                                                                                <TableCell colSpan={2}>
                                                                                    <Typography variant='body2' color='textSecondary'>
                                                                                        No metrics available for this run.
                                                                                    </Typography>
                                                                                </TableCell>
                                                                            </TableRow>
                                                                        )}
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                Trained
                                                                                Weights
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                <DisplayJson
                                                                                    data={
                                                                                        selectedRun
                                                                                            .response_data
                                                                                            .trainData
                                                                                    }
                                                                                />
                                                                            </TableCell>
                                                                        </TableRow>

                                                                        {/* File Names */}
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                Hmod
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                {
                                                                                    selectedRun.file1_name
                                                                                }
                                                                            </TableCell>
                                                                        </TableRow>
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                CSV
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                {
                                                                                    selectedRun.file2_name
                                                                                }
                                                                            </TableCell>
                                                                        </TableRow>
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                NewHmod
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                {
                                                                                    selectedRun
                                                                                        .response_data
                                                                                        .new_hmod
                                                                                }
                                                                            </TableCell>
                                                                        </TableRow>

                                                                        {/* Mode */}
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                Mode
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                {
                                                                                    mode
                                                                                }
                                                                            </TableCell>
                                                                        </TableRow>

                                                                        {/* Created At */}
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                Created
                                                                                At
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                {selectedRun.createdAt
                                                                                    ? selectedRun.createdAt
                                                                                          .toDate()
                                                                                          .toLocaleString()
                                                                                    : "N/A"}
                                                                            </TableCell>
                                                                        </TableRow>

                                                                        {/* Finished At */}
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                Finished
                                                                                At
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                {selectedRun.finishedAt
                                                                                    ? selectedRun.finishedAt
                                                                                          .toDate()
                                                                                          .toLocaleString()
                                                                                    : "Not Finished"}
                                                                            </TableCell>
                                                                        </TableRow>

                                                                        {/* Duration */}
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                Duration
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                {selectedRun.createdAt &&
                                                                                selectedRun.finishedAt
                                                                                    ? calculateDuration(
                                                                                          selectedRun.createdAt,
                                                                                          selectedRun.finishedAt
                                                                                      )
                                                                                    : "N/A"}
                                                                            </TableCell>
                                                                        </TableRow>

                                                                        <TableRow>
                                                                            <TableCell
                                                                                colSpan={
                                                                                    2
                                                                                }
                                                                                sx={{
                                                                                    backgroundColor:
                                                                                        "#f5f5f5",
                                                                                }}>
                                                                                <Typography variant='h6'>
                                                                                    Machine
                                                                                    Learning
                                                                                    Options
                                                                                </Typography>
                                                                            </TableCell>
                                                                        </TableRow>

                                                                        {/* Inputs and Outputs */}
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                Inputs
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                {selectedRun.Inputs ??
                                                                                    "N/A"}
                                                                            </TableCell>
                                                                        </TableRow>
                                                                        <TableRow>
                                                                            <TableCell
                                                                                component='th'
                                                                                scope='row'>
                                                                                Outputs
                                                                            </TableCell>
                                                                            <TableCell>
                                                                                {selectedRun.Outputs ??
                                                                                    "N/A"}
                                                                            </TableCell>
                                                                        </TableRow>

                                                                        {/* Machine Learning Options */}
                                                                        {selectedRun.MachineLearning &&
                                                                            machineLearningOptions.map(
                                                                                ({
                                                                                    key,
                                                                                    label,
                                                                                }) => {
                                                                                    const value =
                                                                                        selectedRun
                                                                                            .MachineLearning[
                                                                                            key
                                                                                        ];
                                                                                    if (
                                                                                        value ===
                                                                                        undefined
                                                                                    ) {
                                                                                        return null;
                                                                                    }
                                                                                    return (
                                                                                        <TableRow
                                                                                            key={
                                                                                                key
                                                                                            }>
                                                                                            <TableCell
                                                                                                component='th'
                                                                                                scope='row'>
                                                                                                {
                                                                                                    label
                                                                                                }
                                                                                            </TableCell>
                                                                                            <TableCell>
                                                                                                {getDisplayValue(
                                                                                                    key,
                                                                                                    value
                                                                                                )}
                                                                                            </TableCell>
                                                                                        </TableRow>
                                                                                    );
                                                                                }
                                                                            )}
                                                                    </TableBody>
                                                                </Table>
                                                            </TableContainer>

                                                            {/* Download Buttons */}
                                                            <div
                                                                style={{
                                                                    marginTop: 20,
                                                                }}>
                                                                <Button
                                                                    variant='contained'
                                                                    href={
                                                                        fileUrls.file1_url
                                                                    }
                                                                    target='_blank'
                                                                    download
                                                                    style={{
                                                                        margin: "10px",
                                                                    }}>
                                                                    Download
                                                                    HMOD
                                                                </Button>
                                                                <Button
                                                                    variant='contained'
                                                                    href={
                                                                        fileUrls.file2_url
                                                                    }
                                                                    target='_blank'
                                                                    download
                                                                    style={{
                                                                        margin: "10px",
                                                                    }}>
                                                                    Download CSV
                                                                </Button>
                                                                <Button
                                                                    variant='contained'
                                                                    href={
                                                                        fileUrls.new_hmod_url
                                                                    }
                                                                    target='_blank'
                                                                    download
                                                                    style={{
                                                                        margin: "10px",
                                                                    }}>
                                                                    Download
                                                                    Trained HMOD
                                                                </Button>
                                                            </div>
                                                        </>
                                                    ) : (
                                                        <Typography color='error'>
                                                            Error: No response
                                                            data available
                                                        </Typography>
                                                    )}
                                                </div>
                                            )}
                                        </Box>
                                    </Modal>
                                    <Modal
                                        open={selectedPlotIndex !== null}
                                        onClose={() =>
                                            setSelectedPlotIndex(null)
                                        }
                                        aria-labelledby='modal-plot-title'
                                        aria-describedby='modal-plot-description'>
                                        <Box
                                            sx={{
                                                position: "absolute",
                                                top: "50%",
                                                left: "50%",
                                                transform:
                                                    "translate(-50%, -50%)",
                                                width: "80%",
                                                height: "90%",
                                                bgcolor: "background.paper",
                                                border: "2px solid #000",
                                                boxShadow: 24,
                                                p: 4,
                                                overflow: "auto",
                                                alignContent: "center",
                                                justifyContent: "center",
                                                display: "flex",
                                                alignItems: "center",
                                                position: "relative",
                                            }}>
                                            <IconButton
                                                onClick={() =>
                                                    setSelectedPlotIndex(null)
                                                }
                                                sx={{
                                                    position: "absolute",
                                                    top: "5%",
                                                    right: "5%",
                                                }}>
                                                <CloseIcon
                                                    sx={{
                                                        height: 50,
                                                        width: 50,
                                                    }}
                                                />
                                            </IconButton>
                                            {selectedPlotIndex !== null && (
                                                <>
                                                    <IconButton
                                                        onClick={handlePrevPlot}
                                                        sx={{
                                                            position:
                                                                "absolute",
                                                            top: "50%",
                                                            left: "16px",
                                                            transform:
                                                                "translateY(-50%)",
                                                            zIndex: 1,
                                                        }}>
                                                        <ArrowBackIcon />
                                                    </IconButton>
                                                    <img
                                                        src={
                                                            selectedRun.plots[
                                                                selectedPlotIndex
                                                            ]
                                                        }
                                                        alt={`Plot ${selectedPlotIndex}`}
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
                                                            position:
                                                                "absolute",
                                                            top: "50%",
                                                            right: "16px",
                                                            transform:
                                                                "translateY(-50%)",
                                                            zIndex: 1,
                                                        }}>
                                                        <ArrowForwardIcon />
                                                    </IconButton>
                                                </>
                                            )}
                                        </Box>
                                    </Modal>
                                    <Dialog
                                        open={deleteDialogOpen}
                                        onClose={handleDeleteDialogClose}
                                        aria-labelledby='alert-dialog-title'
                                        aria-describedby='alert-dialog-description'>
                                        <DialogTitle id='alert-dialog-title'>
                                            {"Delete Run"}
                                        </DialogTitle>
                                        <DialogContent>
                                            <DialogContentText id='alert-dialog-description'>
                                                Are you sure you want to delete:{" "}
                                                {runToDelete
                                                    ? runToDelete.description
                                                    : ""}
                                                ? This action cannot be undone.
                                            </DialogContentText>
                                        </DialogContent>
                                        <DialogActions>
                                            <Button
                                                onClick={
                                                    handleDeleteDialogClose
                                                }
                                                color='primary'>
                                                Cancel
                                            </Button>
                                            <Button
                                                onClick={deleteRun}
                                                color='primary'
                                                autoFocus
                                                disabled={isloading}>
                                                {isloading
                                                    ? "Deleting..."
                                                    : "Delete"}
                                            </Button>
                                        </DialogActions>
                                    </Dialog>
                                </Paper>
                            </Grid>
                        </Grid>
                    </Container>
                    <footer
                        style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            padding: "1em",
                            background: "#f1f1f1",
                            width: "100%",
                            marginTop: "auto",
                        }}>
                        
                        <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
                            <p style={{ margin: 0, textAlign: "center" }}>
                            &copy; {new Date().getFullYear()} NOVA School of Science and Technology,
                            Universidade NOVA de Lisboa. All rights reserved.
                            </p>

                            <Button
                            color="inherit"
                            variant="text"
                            onClick={handleContactUsClick}
                            style={{
                                marginTop: "0.5em",
                                alignSelf: "center",
                                textTransform: "none",
                            }}
                            startIcon={<EmailIcon />}
                            >
                            Contact Us
                            </Button>
                        </div>

                        <a
                            href="https://www.fct.unl.pt/en"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <img
                            src="https://www.fct.unl.pt/sites/default/files/images/logo_nova_fct_pt_v.png"
                            width="75px"
                            alt="FCT Logo"
                            style={{ marginLeft: "1em" }}
                            />
                        </a>
                    </footer>
                </Box>
            </Box>
        </ThemeProvider>
    );
}
