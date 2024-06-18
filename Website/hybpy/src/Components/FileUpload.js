import React, { useState, useEffect } from "react";
import { auth } from "../firebase-config";
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
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import { mainListItems, secondaryListItems } from "./ListItems";
import { useNavigate } from "react-router-dom";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import PublishIcon from "@mui/icons-material/Publish";
import Button from "@mui/material/Button";
import {
    Input,
    Select,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Tabs,
    Tab,
    MenuItem,
    Tooltip,
    tooltipClasses,
    Link,
    TextField,
    Modal,
} from "@mui/material";
import * as XLSX from "xlsx";
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";
import logo from "../Image/HYBpyINVIS_logo.png";
import { LineChart } from "./LineChart";
import InfoIcon from "@mui/icons-material/Info";

const drawerWidth = 200;

const VisuallyHiddenInput = styled("input")`
    clip: rect(0 0 0 0);
    clip-path: inset(50%);
    height: 1px;
    overflow: hidden;
    position: absolute;
    bottom: 0;
    left: 0;
    white-space: nowrap;
    width: 1px;
`;

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
const extractValue = (content, key, defaultValue) => {
    const regex = new RegExp(`${key}=([^;]+);`);
    const match = content.match(regex);
    return match ? match[1].trim() : defaultValue;
};

const TrainingModal = ({ open, handleClose }) => (
    <Dialog open={open} onClose={handleClose}>
        <DialogTitle>Training in Progress</DialogTitle>
        <DialogContent>
            <Grid container spacing={2} marginTop={2}>
                <Grid item xs={12}>
                    <Typography id='modal-modal-title' variant='h6' component='h2'>
                        Training in Progress
                    </Typography>
                </Grid>
                <Grid item xs={12}>
                    <Typography id='modal-modal-description' sx={{ mt: 2 }}>
                        Your training has started. You will be redirected to the dashboard once you
                        close this modal.
                    </Typography>
                </Grid>
            </Grid>
        </DialogContent>
        <DialogActions>
            <Button onClick={handleClose} color='primary'>
                Close
            </Button>
        </DialogActions>
    </Dialog>
);

function FileUpload() {
    const navigate = useNavigate();

    const navigateToUpload = () => {
        navigate("/Dashboard");
    };
    const [open, setOpen] = useState(true);
    const toggleDrawer = () => {
        setOpen(!open);
    };
    const [selectedFile1, setSelectedFile1] = useState(null);
    const [file1Content, setFile1Content] = useState("");
    const [selectedFile2, setSelectedFile2] = useState(null);
    const [file2Content, setFile2Content] = useState([]);
    const [mode, setMode] = useState("");
    const [backendResponse, setBackendResponse] = useState("");
    const [description, setDescription] = useState("");
    const [tooltipDisplay, setTooltipDisplay] = useState("block");
    const [modalOpen, setModalOpen] = useState(false);
    const [tabIndex, setTabIndex] = useState(0);
    const [batchData, setBatchData] = useState([]);
    const [progress, setProgress] = useState(0);
    const [hmodModalOpen, setHmodModalOpen] = useState(false);
    const [initialValues, setInitialValues] = useState(null);
    const [trainingModalOpen, setTrainingModalOpen] = useState(false);
    const [runInProgress, setRunInProgress] = useState(false);

    const handleCloseTrainingModal = () => {
        setTrainingModalOpen(false);
        navigate("/Dashboard");
    };

    const handleFileChange1 = (event) => {
        setSelectedFile1(event.target.files[0]);
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                const content = e.target.result;
                setFile1Content(content);

                const regexPrefix = /(\w+)\.nspecies=/g;
                const uniquePrefixes = new Set();
                let match;
                while ((match = regexPrefix.exec(content)) !== null) {
                    uniquePrefixes.add(match[1]);
                }

                let initialValues = {};
                uniquePrefixes.forEach((prefix) => {
                    const hiddenNodesMatch = content.match(
                        new RegExp(`${prefix}\\.mlm\\.options=\\{'hidden nodes', \\[(.*?)\\]\\};`)
                    );
                    const hiddenNodes = hiddenNodesMatch
                        ? hiddenNodesMatch[1].replace(/,/g, " ")
                        : "5 5";

                    initialValues = {
                        ...initialValues,
                        hiddenNodes,
                        layer: parseFloat(extractValue(content, `${prefix}.mlm.layer`)),
                        tau: parseFloat(extractValue(content, `${prefix}.time.TAU`)),
                        mode: parseFloat(extractValue(content, `${prefix}.mode`)),
                        method: parseFloat(extractValue(content, `${prefix}.method`)),
                        jacobian: parseFloat(extractValue(content, `${prefix}.jacobian`)),
                        hessian: parseFloat(extractValue(content, `${prefix}.hessian`)),
                        niter: parseFloat(extractValue(content, `${prefix}.niter`)),
                        nstep: parseFloat(extractValue(content, `${prefix}.nstep`)),
                        bootstrap: parseFloat(extractValue(content, `${prefix}.bootstrap`)),
                    };
                });

                setInitialValues(initialValues);
            };
            reader.readAsText(file);
        }
        setProgress(3);
    };

    const handleFileChange2 = (event) => {
        setSelectedFile2(event.target.files[0]);
        const file = event.target.files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
            const workbook = XLSX.read(e.target.result, { type: "binary" });
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];
            const data = XLSX.utils.sheet_to_json(worksheet, { defval: "" });

            const separatedBatches = [];
            let currentBatch = [];

            data.forEach((row, index) => {
                const timeValue = row["time"];
                const nextTimeValue = index + 1 < data.length ? data[index + 1]["time"] : null;

                if (nextTimeValue !== null && nextTimeValue < timeValue) {
                    currentBatch.push(row);
                    separatedBatches.push(currentBatch);
                    currentBatch = [];
                } else {
                    currentBatch.push(row);
                }
            });

            if (currentBatch.length > 0) {
                separatedBatches.push(currentBatch);
            }

            setFile2Content(data);
            setBatchData(separatedBatches);
        };
        reader.readAsBinaryString(file);
        setProgress(2);
    };

    const handleModeChange = (event) => {
        setMode(event.target.value);
        if (event.target.value === "1") {
            setProgress(4);
        } else if (event.target.value === "2") {
            setProgress(5);
        }
    };

    const changeTooltip = () => {
        setTooltipDisplay(tooltipDisplay === "block" ? "none" : "block");
    };

    const CustomWidthTooltip = styled(({ className, tooltip, ...props }) => (
        <Tooltip {...props} classes={{ popper: className }} />
    ))(({}) => ({
        [`& .${tooltipClasses.tooltip}`]: {
            maxWidth: 200,
        },
    }));

    const handleUpload = async () => {
        if (!selectedFile1 || !selectedFile2) {
            alert("Please select both files!");
            return;
        }

        if (mode !== "1" && mode !== "2") {
            alert("Please select a mode (1 or 2)!");
            return;
        }

        const formData = new FormData();
        formData.append("file1", selectedFile1);
        formData.append("file2", selectedFile2);
        formData.append("mode", mode);
        formData.append("userId", auth.currentUser.uid);
        formData.append("description", description);
        formData.append("train_batches", Array.from(train_batches).join(","));
        formData.append("test_batches", Array.from(test_batches).join(","));
        formData.append("user_id", auth.currentUser.uid);

        setTrainingModalOpen(true); // Open the modal when the upload starts

        try {
            const response = await fetch("http://localhost:5000/upload", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            setBackendResponse(JSON.stringify(data, null, 2));

            checkRunStatus(); // Start checking the run status
        } catch (error) {
            console.error("Error uploading file:", error);
            setBackendResponse(`Error: ${error.message}`);
        }
    };

    const checkRunStatus = async () => {
        const userId = auth.currentUser.uid;
        const intervalId = setInterval(async () => {
            try {
                const response = await fetch(`http://localhost:5000/run-status?user_id=${userId}`);
                const data = await response.json();
                if (data.status === "no_runs") {
                    setRunInProgress(false);
                } else if (data.status === "in_progress") {
                    setRunInProgress(true);
                } else {
                    clearInterval(intervalId);
                    setRunInProgress(false);
                    navigate("/Dashboard");
                }
            } catch (error) {
                console.error("Error checking run status:", error);
            }
        }, 5000);
    };

    const [availableBatches, setAvailableBatches] = useState([]);
    const [train_batches, setTrainBatches] = useState(new Set());
    const [test_batches, setTestBatches] = useState(new Set());

    useEffect(() => {
        const fetchAvailableBatches = async () => {
            if (selectedFile2 && mode === "1") {
                try {
                    const formData = new FormData();
                    formData.append("file2", selectedFile2);

                    const response = await fetch("http://localhost:5000/get-available-batches", {
                        method: "POST",
                        body: formData,
                    });

                    const data = await response.json();
                    setAvailableBatches(data);
                } catch (error) {
                    console.error("Error fetching batches:", error);
                }
            }
        };

        fetchAvailableBatches();
    }, [selectedFile2, mode]);

    const handleTrainBatchSelection = (batch) => {
        setTrainBatches((prevSelectedBatches) => {
            const newSelection = new Set(prevSelectedBatches);
            if (newSelection.has(batch)) {
                newSelection.delete(batch);
            } else if (!test_batches.has(batch)) {
                newSelection.add(batch);
            }
            return newSelection;
        });
    };

    const handleTestBatchSelection = (batch) => {
        setTestBatches((prevSelectedBatches) => {
            const newSelection = new Set(prevSelectedBatches);
            if (newSelection.has(batch)) {
                newSelection.delete(batch);
            } else if (!train_batches.has(batch)) {
                newSelection.add(batch);
            }
            return newSelection;
        });
    };

    const handleTabChange = (event, newValue) => {
        setTabIndex(newValue);
    };

    const handleHmodModalSave = (updatedOptions) => {
        setHmodModalOpen(false);

        let updatedContent = file1Content;

        const regexPrefix = /(\w+)\.nspecies=/g; // Match the prefix of the HMOD file (e.g. 'model' in 'model.nspecies=3;')
        // Get all unique prefixes in the HMOD file (e.g. 'model') and store them in a Set to avoid duplicates
        const uniquePrefixes = new Set();
        let match;
        while ((match = regexPrefix.exec(file1Content)) !== null) {
            uniquePrefixes.add(match[1]);
        }

        // Create a map of options to update in the HMOD file based on the unique prefixes found in the file. This can be extended to include more options.
        const optionsMap = {};
        uniquePrefixes.forEach((prefix) => {
            optionsMap[
                `${prefix}\\.mlm\\.options`
            ] = `${prefix}.mlm.options={'hidden nodes', [${updatedOptions.hiddenNodes.join(
                " "
            )}]};`;
            optionsMap[`${prefix}\\.mlm\\.layer`] = `${prefix}.mlm.layer=${updatedOptions.layer};`;
            optionsMap[`${prefix}\\.time\\.TAU`] = `${prefix}.time.TAU=${updatedOptions.tau};`;
            optionsMap[`${prefix}\\.mode`] = `${prefix}.mode=${updatedOptions.mode};`;
            optionsMap[`${prefix}\\.method`] = `${prefix}.method=${updatedOptions.method};`;
            optionsMap[`${prefix}\\.jacobian`] = `${prefix}.jacobian=${updatedOptions.jacobian};`;
            optionsMap[`${prefix}\\.hessian`] = `${prefix}.hessian=${updatedOptions.hessian};`;
            optionsMap[`${prefix}\\.niter`] = `${prefix}.niter=${updatedOptions.niter};`;
            optionsMap[`${prefix}\\.nstep`] = `${prefix}.nstep=${updatedOptions.nstep};`;
            optionsMap[
                `${prefix}\\.bootstrap`
            ] = `${prefix}.bootstrap=${updatedOptions.bootstrap};`;
        });

        for (const [key, value] of Object.entries(optionsMap)) {
            const regex = new RegExp(`${key}=.*?;`, "g");
            updatedContent = updatedContent.replace(regex, value);
        }

        setFile1Content(updatedContent);

        const updatedFile = new Blob([updatedContent], { type: "text/plain" });
        const updatedFileObject = new File([updatedFile], selectedFile1.name, {
            type: selectedFile1.type,
        });
        setSelectedFile1(updatedFileObject);
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
                    <Container maxWidth='lg' sx={{}}>
                        <div style={{ overflow: "auto", marginTop: 20 }}>
                            <h2 style={{ float: "left", marginTop: 0 }}>New Hybrid Model</h2>
                        </div>
                        <Grid container spacing={3} columns={20}>
                            <Grid item xs={20}>
                                <Paper sx={{ p: 2, display: "flex", flexDirection: "column" }}>
                                    <div style={{ display: "flex", alignItems: "center" }}>
                                        <Typography variant='h5'>Title</Typography>
                                        <Tooltip
                                            title='In this section you can write the Title of the Model you are going to create.'
                                            arrow>
                                            <IconButton size='small' sx={{ ml: 1 }}>
                                                <InfoIcon />
                                            </IconButton>
                                        </Tooltip>
                                    </div>
                                    <Input
                                        fullWidth
                                        value={description}
                                        onChange={(e) => {
                                            setDescription(e.target.value);
                                            if (e.target.value) {
                                                setProgress(1);
                                            } else {
                                                setProgress(0);
                                            }
                                        }}
                                    />
                                </Paper>
                            </Grid>

                            <Grid item xs={20} columns={20}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        height: 400,
                                        overflow: "auto",
                                    }}>
                                    <div style={{ display: "flex", alignItems: "center" }}>
                                        <Typography variant='h5'>CSV</Typography>
                                        <Tooltip
                                            title='In this we will ask you to upload the CSV file which is a file containing the information about the batches. After uploading there will be a preview of the file.'
                                            arrow>
                                            <IconButton size='small' sx={{ ml: 1 }}>
                                                <InfoIcon />
                                            </IconButton>
                                        </Tooltip>
                                    </div>
                                    <Typography variant='h7'>
                                        {" "}
                                        Step 1: Please select Dataset (.csv). See{" "}
                                        <Link>Template</Link> and <Link>Example</Link>.{" "}
                                    </Typography>
                                    <p>
                                        {selectedFile2
                                            ? selectedFile2.name
                                            : "Insert your CSV file containing the information about the batches here."}
                                    </p>
                                    <TableContainer
                                        component={Paper}
                                        sx={{ maxHeight: 240, overflow: "auto", fontSize: 1 }}>
                                        <Table size='small' aria-label='a dense table'>
                                            <TableHead>
                                                <TableRow>
                                                    {file2Content.length > 0 &&
                                                        Object.keys(file2Content[0]).map((key) => (
                                                            <TableCell key={key}>{key}</TableCell>
                                                        ))}
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {file2Content.map((row, idx) => (
                                                    <TableRow key={idx}>
                                                        {Object.keys(file2Content[0]).map((key) => (
                                                            <TableCell key={idx + key}>
                                                                {row[key]}
                                                            </TableCell>
                                                        ))}
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </Paper>
                                <div style={{ display: "flex", marginTop: "8px" }}>
                                    <label htmlFor='csv-upload' style={{ flex: 1 }}>
                                        <Button
                                            fullWidth
                                            variant='contained'
                                            sx={{ height: "100%", marginBottom: 2 }}
                                            disabled={progress < 1}
                                            component='span'>
                                            <PublishIcon fontSize='large' />
                                            Upload CSV
                                        </Button>
                                        <VisuallyHiddenInput
                                            type='file'
                                            id='csv-upload'
                                            disabled={progress < 1}
                                            onChange={handleFileChange2}
                                        />
                                    </label>
                                    <Button
                                        onClick={() => setModalOpen(true)}
                                        variant='contained'
                                        sx={{
                                            marginLeft: "16px",
                                            height: "100%",
                                            marginBottom: 2,
                                        }}
                                        disabled={progress < 2}>
                                        View Batches
                                    </Button>
                                </div>
                                <Dialog
                                    open={modalOpen}
                                    onClose={() => setModalOpen(false)}
                                    maxWidth='lg'
                                    fullWidth>
                                    <DialogTitle>Batch Data Visualization</DialogTitle>
                                    <DialogContent>
                                        <Tabs
                                            value={tabIndex}
                                            onChange={handleTabChange}
                                            indicatorColor='primary'
                                            textColor='primary'
                                            variant='scrollable'
                                            scrollButtons='auto'>
                                            {batchData.map((batch, index) => (
                                                <Tab key={index} label={`Batch ${index + 1}`} />
                                            ))}
                                        </Tabs>
                                        <DialogContent>
                                            {batchData.map((batch, index) => (
                                                <div
                                                    key={index}
                                                    style={{
                                                        display:
                                                            tabIndex === index ? "block" : "none",
                                                    }}>
                                                    <LineChart data={batch} />
                                                </div>
                                            ))}
                                        </DialogContent>
                                    </DialogContent>
                                    <DialogActions>
                                        <Button onClick={() => setModalOpen(false)} color='primary'>
                                            Close
                                        </Button>
                                    </DialogActions>
                                </Dialog>
                            </Grid>

                            <Grid item xs={20} columns={20}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        height: 400,
                                        overflow: "auto",
                                    }}>
                                    <div style={{ display: "flex", alignItems: "center" }}>
                                        <Typography variant='h5'>HMOD</Typography>
                                        <Tooltip
                                            title='Insert your HMOD file containing the information about the mechanistic model and the settings for the machine learning model here.'
                                            arrow>
                                            <IconButton size='small' sx={{ ml: 1 }}>
                                                <InfoIcon />
                                            </IconButton>
                                        </Tooltip>
                                    </div>
                                    <Typography variant='h7'>
                                        Step 2. Please select HMOD hybrid/standard model (.hmod).
                                        The HMOD file is an intermediate format that enables
                                        communication between the essential components of the
                                        mechanistic and hybrid models. You can use the{" "}
                                        <Link>SBML2HYB</Link> tool to create a hybrid HMOD model
                                        format from any SBML model or see <Link> Example 1</Link>{" "}
                                        and <Link> Example 2</Link> to edit/create your own standard
                                        HMOD model.
                                    </Typography>
                                    <pre>{file1Content}</pre>{" "}
                                </Paper>
                                <div style={{ display: "flex", marginTop: "8px" }}>
                                    <label htmlFor='hmod-upload' style={{ flex: 1 }}>
                                        <Button
                                            component='span'
                                            fullWidth
                                            variant='contained'
                                            sx={{ height: "100%", marginBottom: 2 }}
                                            disabled={progress < 2}>
                                            <PublishIcon fontSize='large' />
                                            Upload Hmod
                                        </Button>
                                        <VisuallyHiddenInput
                                            type='file'
                                            id='hmod-upload'
                                            disabled={progress < 2}
                                            onChange={handleFileChange1}
                                        />
                                    </label>
                                    <Button
                                        variant='contained'
                                        onClick={() => setHmodModalOpen(true)}
                                        disabled={progress < 3}
                                        sx={{
                                            marginLeft: "16px",
                                            height: "100%",
                                            marginBottom: 2,
                                        }}>
                                        Edit HMOD Options
                                    </Button>
                                </div>
                            </Grid>

                            <Grid item xs={7}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        marginBottom: 2,
                                    }}>
                                    <div style={{ display: "flex", alignItems: "center" }}>
                                        <Typography variant='h5'>Mode</Typography>
                                        <Tooltip
                                            title='In this section you can select the batch selection mode. 1 is for selecting train and test batches manually from a list and 2 is for the selection to be done randomly (with a 2/3; 1/3 split).'
                                            arrow>
                                            <IconButton size='small' sx={{ ml: 1 }}>
                                                <InfoIcon />
                                            </IconButton>
                                        </Tooltip>
                                    </div>
                                    <Select
                                        labelId='Mode'
                                        id='Mode'
                                        value={mode}
                                        style={{ marginTop: 20 }}
                                        onChange={handleModeChange}
                                        disabled={progress < 3}>
                                        <MenuItem value={"1"}>1</MenuItem>
                                        <MenuItem value={"2"}>2</MenuItem>
                                    </Select>
                                </Paper>
                            </Grid>
                            {(mode === "1" && progress >= 4 && (
                                <>
                                    <Grid item xs={3}>
                                        <h3>Available Batches: {availableBatches.join(", ")}</h3>
                                    </Grid>
                                    <Grid item xs={3}>
                                        <h3>Select Train Batches: </h3>
                                        {availableBatches.map((batch) => (
                                            <div key={batch}>
                                                <input
                                                    type='checkbox'
                                                    checked={Array.from(train_batches).includes(
                                                        batch
                                                    )}
                                                    onChange={() =>
                                                        handleTrainBatchSelection(batch)
                                                    }
                                                />
                                                {batch}
                                            </div>
                                        ))}
                                    </Grid>
                                    <Grid item xs={3}>
                                        <h3>Select Test Batches: </h3>
                                        {availableBatches.map((batch) => (
                                            <div key={batch}>
                                                <input
                                                    type='checkbox'
                                                    checked={Array.from(test_batches).includes(
                                                        batch
                                                    )}
                                                    onChange={() => handleTestBatchSelection(batch)}
                                                />
                                                {batch}
                                            </div>
                                        ))}
                                    </Grid>
                                    <Grid item xs={4}>
                                        <Button
                                            onClick={() => handleUpload()}
                                            fullWidth
                                            variant='contained'
                                            sx={{ height: "100%" }}
                                            disabled={
                                                train_batches.size === 0 || test_batches.size === 0
                                            }>
                                            <PublishIcon fontSize='large' />
                                            Upload Information
                                        </Button>
                                    </Grid>
                                </>
                            )) ||
                                (mode === "2" && progress >= 5 && (
                                    <>
                                        <Grid item xs={4}></Grid>
                                        <CustomWidthTooltip
                                            title={
                                                tooltipDisplay === "block"
                                                    ? "After clicking on the Upload Information button the information will be uploaded and the run will be created."
                                                    : ""
                                            }
                                            followCursor
                                            arrow>
                                            <Grid item xs={6}>
                                                <Button
                                                    onClick={() => handleUpload()}
                                                    fullWidth
                                                    variant='contained'
                                                    sx={{ height: "100%" }}>
                                                    <PublishIcon fontSize='large' />
                                                    Upload Information
                                                </Button>
                                            </Grid>
                                        </CustomWidthTooltip>
                                    </>
                                ))}
                        </Grid>
                    </Container>
                </Box>
            </Box>
            <TrainingModal open={trainingModalOpen} handleClose={handleCloseTrainingModal} />
        </ThemeProvider>
    );
}

export default FileUpload;
