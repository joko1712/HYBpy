import React, { useState } from "react";
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
import { Input, Select } from "@mui/material";
import MenuItem from "@mui/material/MenuItem";
import * as XLSX from "xlsx";
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";

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

function FileUpload() {
    const navigate = useNavigate();

    const navigateToUpload = () => {
        navigate("/");
    };
    const [open, setOpen] = React.useState(true);
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

    const handleFileChange1 = (event) => {
        setSelectedFile1(event.target.files[0]);
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                setFile1Content(e.target.result);
            };
            reader.readAsText(file);
        }
    };

    const handleFileChange2 = (event) => {
        setSelectedFile2(event.target.files[0]);
        const file = event.target.files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
            const workbook = XLSX.read(e.target.result, { type: "binary" });
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];
            const data = XLSX.utils.sheet_to_json(worksheet);
            setFile2Content(Array.isArray(data) ? data : []);
        };
        reader.readAsBinaryString(file);
    };

    const handleModeChange = (event) => {
        setMode(event.target.value);
    };

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

        try {
            const response = await fetch("http://localhost:5000/upload", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            setBackendResponse(JSON.stringify(data, null, 2)); // Store the response data in state
        } catch (error) {
            console.error("Error uploading file:", error);
            setBackendResponse(`Error: ${error.message}`); // Store error message in state
        }
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
                            <IconButton color='inherit' onClick={() => navigateToUpload()}>
                                Dashboard
                            </IconButton>
                        </Typography>
                        {/*Check if there are any runs in progress if so display progress bar if not display nothing */}
                        <IconButton color='inherit' size='small'>
                            <p>Run Progress:</p>
                        </IconButton>
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
                        height: "100vh",
                        overflow: "auto",
                    }}>
                    <Toolbar />
                    <Container maxWidth='lg' sx={{ mt: 1, mb: 4 }}>
                        <h2>Create Run</h2>
                        <Grid container spacing={3} columns={20}>
                            <Grid item xs={10} columns={10}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        height: 240,
                                        overflow: "auto",
                                    }}>
                                    <p>Hmod</p>
                                    <p>{selectedFile1 ? selectedFile1.name : "No file selected"}</p>
                                    <pre>{file1Content}</pre>{" "}
                                </Paper>
                                <label htmlFor='hmod-upload'>
                                    <Grid item xs={10}>
                                        <Button
                                            component='span'
                                            fullWidth
                                            variant='contained'
                                            sx={{ height: "100%" }}>
                                            <PublishIcon fontSize='large' />
                                            Upload Hmod
                                        </Button>
                                        <VisuallyHiddenInput
                                            type='file'
                                            id='hmod-upload'
                                            onChange={handleFileChange1}
                                        />
                                    </Grid>
                                </label>
                            </Grid>
                            <Grid item xs={10} columns={10}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        height: 240,
                                        overflow: "auto",
                                    }}>
                                    <p>CSV</p>
                                    <p>{selectedFile2 ? selectedFile2.name : "No file selected"}</p>
                                    <TableContainer
                                        component={Paper}
                                        sx={{ maxHeight: 240, overflow: "auto" }}>
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
                                <label htmlFor='csv-upload'>
                                    <Grid item xs={10}>
                                        <Button
                                            component='span'
                                            fullWidth
                                            variant='contained'
                                            sx={{ height: "100%" }}>
                                            <PublishIcon fontSize='large' />
                                            Upload Hmod
                                        </Button>
                                        <VisuallyHiddenInput
                                            type='file'
                                            id='csv-upload'
                                            onChange={handleFileChange2}
                                        />
                                    </Grid>
                                </label>
                            </Grid>
                            <Grid item xs={20}>
                                <Paper sx={{ p: 2, display: "flex", flexDirection: "column" }}>
                                    <p>Description</p>
                                    <Input onChange={(e) => setDescription(e.target.value)}></Input>
                                </Paper>
                            </Grid>
                            <Grid item xs={13}>
                                <Paper sx={{ p: 2, display: "flex", flexDirection: "column" }}>
                                    <p>Mode:</p>
                                    <Select
                                        labelId='Mode'
                                        id='Mode'
                                        value={mode}
                                        label='Mode'
                                        onChange={(e) => setMode(e.target.value)}>
                                        <MenuItem value={"1"}>1</MenuItem>
                                        <MenuItem value={"2"}>2</MenuItem>
                                    </Select>
                                </Paper>
                            </Grid>
                            <Grid item xs={7}>
                                <Button
                                    onClick={() => handleUpload()}
                                    fullWidth
                                    variant='contained'
                                    sx={{ height: "100%" }}>
                                    <PublishIcon fontSize='large' />
                                    Upload Imformation
                                </Button>
                            </Grid>
                        </Grid>
                    </Container>
                </Box>
            </Box>
        </ThemeProvider>
    );
}

export default FileUpload;
