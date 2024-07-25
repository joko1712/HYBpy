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
} from "@mui/material";

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
                        <th key={key} style={{ border: "1px solid black", padding: "5px" }}>
                            {key}
                        </th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {data.map((item, index) => (
                    <tr key={index}>
                        {Object.values(item).map((value, i) => (
                            <td key={i} style={{ border: "1px solid black", padding: "5px" }}>
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
    const navigate = useNavigate();

    const navigateToUpload = () => {
        navigate("/Dashboard");
    };
    const [open, setOpen] = React.useState(true);
    const toggleDrawer = () => {
        setOpen(!open);
    };

    const [runs, setRuns] = React.useState([]);
    const [searchQuery, setSearchQuery] = React.useState("");
    const userId = auth.currentUser.uid;
    const [mode, setMode] = React.useState("Error");
    const [date, setDate] = React.useState("Error");

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

    const handlePlotClick = (url) => {
        setSelectedPlot(url);
    };

    const handleDeleteDialogOpen = (run) => {
        setRunToDelete(run);
        setDeleteDialogOpen(true);
    };

    const handleDeleteDialogClose = () => {
        setRunToDelete(null);
        setDeleteDialogOpen(false);
    };

    const fetchFileUrls = async (userId, runId) => {
        const response = await fetch(
            `http://localhost:5000/get-file-urls?user_id=${userId}&run_id=${runId}`
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

            const response = await fetch("http://localhost:5000/delete-run", {
                method: "DELETE",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    user_id: userId,
                    run_id: runToDelete.id,
                    folder_path: folderPath,
                }),
            });
            const result = await response.json();
            console.log("Delete response:", result);
            if (response.ok) {
                setRuns((prevRuns) => prevRuns.filter((run) => run.id !== runToDelete.id));
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
            Method: { 1: "TRF", 2: "Trust-Constr", 3: "Simulated Annealing", 4: "ADAM" },
            Mode: { 1: "Indirect", 2: "Direct", 3: "Indirect" },
            Layer: { 1: "Tanh", 2: "ReLU", 3: "LSTM" },
        };

        return mappings[key] && mappings[key][value] ? mappings[key][value] : value;
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
            if (latestRun.length > 0) {
                setMode(latestRun[0].mode === 1 ? "Manual" : "Automatic");
                setDate(latestRun[0].createdAt.toDate().toLocaleString());
            }
        };

        fetchLatestRun();
    }, [userId]);

    const filteredRuns = runs.filter((run) => {
        return (
            searchQuery === "" ||
            (run.description.toLowerCase().includes(searchQuery.toLowerCase()) &&
                run.file1_name.toLowerCase().includes(searchQuery.toLowerCase())) ||
            run.file2_name.toLowerCase().includes(searchQuery.toLowerCase())
        );
    });

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
                        <h1>
                            <strong>List of Projects</strong>
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
                                                <ListItemText
                                                    primary={`Title: ${run.description}`}
                                                    secondary={`HMOD:${run.file1_name} - CSV:${run.file2_name} - Mode:${mode} - CreatedAt:${date}`}
                                                />

                                                <IconButton
                                                    sx={{ color: red[500] }}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleDeleteDialogOpen(run);
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
                                        <Box sx={{ ...style, width: "80%", height: "90%" }}>
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
                                                            Title: {selectedRun.description}
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
                                                        Model Evaluation: Time Series Prediction
                                                    </Typography>
                                                    <Typography
                                                        id='modal-modal-description'
                                                        sx={{ mt: 3 }}>
                                                        <br />
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
                                                        {selectedRun.response_data ? (
                                                            <>
                                                                Metrics:{" "}
                                                                <pre>
                                                                    {JSON.stringify(
                                                                        selectedRun.response_data
                                                                            .metrics,
                                                                        null,
                                                                        2
                                                                    )}
                                                                </pre>
                                                                <br />
                                                                Trained weights:
                                                                <DisplayJson
                                                                    data={
                                                                        selectedRun.response_data
                                                                            .trainData
                                                                    }
                                                                />
                                                            </>
                                                        ) : (
                                                            <Typography color='error'>
                                                                Error: No response data available
                                                            </Typography>
                                                        )}
                                                        <br />
                                                        Hmod: {selectedRun.file1_name}
                                                        <br />
                                                        CSV: {selectedRun.file2_name}
                                                        <br />
                                                        NewHmod:{" "}
                                                        {selectedRun.response_data.new_hmod}
                                                        <br />
                                                        Mode: {mode}
                                                        <br />
                                                        <Typography variant='h6' marginTop={3}>
                                                            Machine Learning Options:
                                                        </Typography>
                                                        {selectedRun.MachineLearning ? (
                                                            <div style={{ marginLeft: 20 }}>
                                                                {Object.keys(
                                                                    selectedRun.MachineLearning
                                                                ).map((key) => (
                                                                    <div key={key}>
                                                                        <strong>{key}: </strong>
                                                                        {getDisplayValue(
                                                                            key,
                                                                            selectedRun
                                                                                .MachineLearning[
                                                                                key
                                                                            ]
                                                                        )}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        ) : (
                                                            <Typography color='error'>
                                                                Error: No Machine Learning Options
                                                                available
                                                            </Typography>
                                                        )}
                                                        <Button
                                                            variant='contained'
                                                            href={fileUrls.file1_url}
                                                            target='_blank'
                                                            download
                                                            style={{ margin: "10px" }}>
                                                            Download HMOD
                                                        </Button>
                                                        <Button
                                                            variant='contained'
                                                            href={fileUrls.file2_url}
                                                            target='_blank'
                                                            download
                                                            style={{ margin: "10px" }}>
                                                            Download CSV
                                                        </Button>
                                                        <Button
                                                            variant='contained'
                                                            href={fileUrls.new_hmod_url}
                                                            target='_blank'
                                                            download
                                                            style={{ margin: "10px" }}>
                                                            Download New HMOD
                                                        </Button>
                                                    </Typography>
                                                </div>
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
                                                height: "90%",
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
                                                {runToDelete ? runToDelete.description : ""}? This
                                                action cannot be undone.
                                            </DialogContentText>
                                        </DialogContent>
                                        <DialogActions>
                                            <Button
                                                onClick={handleDeleteDialogClose}
                                                color='primary'>
                                                Cancel
                                            </Button>
                                            <Button
                                                onClick={deleteRun}
                                                color='primary'
                                                autoFocus
                                                disabled={isloading}>
                                                {isloading ? "Deleting..." : "Delete"}
                                            </Button>
                                        </DialogActions>
                                    </Dialog>
                                </Paper>
                            </Grid>
                        </Grid>
                    </Container>
                </Box>
            </Box>
        </ThemeProvider>
    );
}
