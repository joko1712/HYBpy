import React, { useState, useEffect, useCallback } from "react";
import { auth } from "../firebase-config";
import { collection, query, where, getDocs, orderBy } from "firebase/firestore";
import { db } from "../firebase-config";
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
    Checkbox,
} from "@mui/material";
import * as XLSX from "xlsx";
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";
import logo from "../Image/HYBpyINVIS_logo_BETA.png";
import { LineChart } from "./LineChart";
import InfoIcon from "@mui/icons-material/Info";
import TrainingModal from "./Modals/TrainingModal";
import HmodModal from "./Modals/HmodModal";
import MlmModal from "./Modals/MlmModal";
import ControlModalSelection from "./Modals/ControlModalSelection";

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

function Simulations() {
    const navigate = useNavigate();
    const navigateToUpload = () => {
        navigate("/");
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
    const [hmodOptions, setHmodOptions] = useState({});
    const [headerModalOpen, setHeaderModalOpen] = useState(false);
    const [selectedHeaders, setSelectedHeaders] = useState([]);
    const [headerModalConfig, setHeaderModalConfig] = useState({
        headers: [],
        onSave: () => {},
        handleClose: () => {},
    });
    const [trainedWeights, setTrainedWeights] = useState("");

    const [mlmModalOpen, setMlmModalOpen] = useState(false);
    const [speciesOptions, setSpeciesOptions] = useState([]);
    const [controlOptions, setControlOptions] = useState([]);
    const [parameterOptions, setParameterOptions] = useState([]);
    const [compartmentOptions, setCompartmentOptions] = useState([]);

    const [mlmOptions, setMlmOptions] = useState({});

    const [completedRuns, setCompletedRuns] = React.useState([]);
    const [selectedRun, setSelectedRun] = React.useState(null);

    const fetchCompletedRuns = async () => {
        const userId = auth.currentUser.uid;
        const runsCollectionRef = collection(db, "users", userId, "runs");
        const q = query(
            runsCollectionRef,
            where("status", "==", "completed"),
            where("userId", "==", userId),
            orderBy("createdAt", "desc")
        );
        const querySnapshot = await getDocs(q);
        const completedRuns = querySnapshot.docs.map((doc) => ({
            id: doc.id,
            ...doc.data(),
        }));
        return completedRuns;
    };

    useEffect(() => {
        const fetchRuns = async () => {
            const runs = await fetchCompletedRuns();
            setCompletedRuns(runs);
        };

        fetchRuns();
    }, []);

    const handleRunSelection = async (run) => {
        setSelectedRun(run);
        console.log("Selected Run: ", run);
        console.log("run.response_data.new_hmod_url: ", run.response_data.new_hmod_url);

        try {
            const response = await fetch(`http://localhost:5000/get-new-hmod`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    url: run.response_data.new_hmod_url,
                }),
            });

            if (!response.ok) {
                throw new Error("Failed to fetch the HMOD file from the server.");
            }

            const blob = await response.blob();

            const fileName = run.response_data.new_hmod || "new_hmod.hmod";
            const file = new File([blob], fileName, { type: "text/plain" });

            const syntheticEvent = {
                target: {
                    files: [file],
                },
            };

            handleFileChange1(syntheticEvent);

            console.log("File downloaded and passed to handleFileChange1.");
        } catch (error) {
            console.error("Error handling the run selection: ", error);
        }
    };

    const handleMlmModalSave = (options) => {
        setMlmModalOpen(false);
        setMlmOptions(options);
    };

    const extractOptionsFromHmod = (content, type) => {
        console.log(`Extracting options for type: ${type}`);
        let regex;
        if (type === "species") {
            regex = /(\w+)\.species\(\d+\)\.id\s*=\s*["']([^"']+)["']/g;
        } else if (type === "control") {
            regex = /(\w+)\.control\(\d+\)\.id\s*=\s*["']([^"']+)["']/g;
        } else if (type === "parameters") {
            regex = /(\w+)\.parameters\(\d+\)\.id\s*=\s*["']([^"']+)["']/g;
        } else if (type === "compartment") {
            regex = /(\w+)\.compartment\(\d+\)\.id\s*=\s*["']([^"']+)["']/g;
        } else {
            return [];
        }

        const options = [];
        let match;
        while ((match = regex.exec(content)) !== null) {
            options.push(match[2]);
        }
        return options;
    };

    const openMlmModal = () => {
        return new Promise((resolve) => {
            const saveMlmHandler = (options) => {
                console.log("MLM Options in saveMlmHandler: ", options);
                setMlmOptions(options);
                resolve(options);
            };

            const closeModalHandler = () => {
                setMlmModalOpen(false);
                resolve({});
            };

            setMlmModalOpen({
                open: true,
                speciesOptions,
                controlOptions,
                parameterOptions,
                compartmentOptions,
                onSave: saveMlmHandler,
                handleClose: closeModalHandler,
            });
        });
    };

    const handleCloseTrainingModal = () => {
        setTrainingModalOpen(false);
        navigate("/results");
    };

    const handleOpenHeaderModal = (headers) => {
        return new Promise((resolve) => {
            const saveHeadersHandler = (headers) => {
                setSelectedHeaders(headers);
                resolve(headers);
            };

            const closeModalHandler = () => {
                setHeaderModalOpen(false);
                resolve([]);
            };

            setHeaderModalOpen(true);
            setHeaderModalConfig({
                headers,
                onSave: saveHeadersHandler,
                handleClose: closeModalHandler,
            });
        });
    };

    const parseWeightsFromHmod = (content) => {
        const prefixMatch = content.match(/(\w+)\.parameters\(\d+\)\.id="w\d+"/);
        if (!prefixMatch) {
            console.error("No prefix found in the HMOD file.");
            return [];
        }

        const dynamicPrefix = prefixMatch[1];
        console.log("Dynamic Prefix Found: ", dynamicPrefix);

        const weightRegex = new RegExp(
            `${dynamicPrefix}\\.parameters\\(\\d+\\)\\.id="w\\d+";\\s*${dynamicPrefix}\\.parameters\\(\\d+\\)\\.val=(-?[\\d\\.eE-]+);`,
            "g"
        );

        let weights = [];
        let match;

        while ((match = weightRegex.exec(content)) !== null) {
            weights.push(parseFloat(match[1]));
        }

        console.log("Extracted Weights: ", weights);

        return weights;
    };

    const extractLayerValue = (content, prefix, defaultValue) => {
        const regex = new RegExp(`${prefix}\\.mlm\\.layer=([0-9]+);`);
        const match = content.match(regex);
        return match ? match[1].trim() : defaultValue;
    };

    const ensureHmodSections = async (
        content,
        batchData,
        headers,
        openHeaderModal,
        selectedHeaders
    ) => {
        console.log("Ensuring HMOD sections...");
        console.log("Content: ", content);
        const regexPrefix = /(\w+)\.nspecies=/g;
        const uniquePrefixes = new Set();

        let match;

        while ((match = regexPrefix.exec(content)) !== null) {
            uniquePrefixes.add(match[1]);
        }

        let maxTime = 0;
        batchData.forEach((row) => {
            if (row.time > maxTime) {
                maxTime = row.time;
            }
        });

        let numBatches = new Set(batchData.map((row) => row.batchId)).size;
        let timeMax = maxTime * numBatches;
        let timeTAU = timeMax / 50;
        let tspan = `0:1:${timeMax}`;

        let updatedContent = content.replace(/\bend\b\s*$/, "");

        if (!updatedContent.includes("time.tspan=")) {
            uniquePrefixes.forEach((prefix) => {
                updatedContent = `${prefix}.time.tspan=${tspan}\n` + updatedContent;
            });
        }

        let controlExists = updatedContent.includes("% control---------------------------");
        let configExists = updatedContent.includes("% %model configuration");
        let mlmExists = updatedContent.includes(
            "% % MLM - Machine Learning Model --------------------------------------------"
        );

        if (headers.length > 0) {
            if (!controlExists) {
                console.log("Control section does not exist, opening header modal...");
                if (selectedHeaders.length === 0) {
                    const newSelectedHeaders = await openHeaderModal(headers);
                    setSelectedHeaders(newSelectedHeaders);
                    selectedHeaders = newSelectedHeaders;
                }
            }
        }

        if (selectedHeaders.length > 0 && !controlExists) {
            let controlSection = `% control---------------------------\n`;
            controlSection += `${uniquePrefixes.values().next().value}.ncontrol=${
                selectedHeaders.length
            };\n`;

            selectedHeaders.forEach((header, index) => {
                let maxHeaderValue = Math.max(...batchData.map((row) => row[header]));
                controlSection += `${uniquePrefixes.values().next().value}.control(${
                    index + 1
                }).id= '${header}';\n`;
                controlSection += `${uniquePrefixes.values().next().value}.control(${
                    index + 1
                }).val= 0;\n`;
                controlSection += `${uniquePrefixes.values().next().value}.control(${
                    index + 1
                }).min= 0;\n`;
                controlSection += `${uniquePrefixes.values().next().value}.control(${
                    index + 1
                }).max= ${maxHeaderValue};\n`;
                controlSection += `${uniquePrefixes.values().next().value}.control(${
                    index + 1
                }).constant= 1;\n`;
            });

            controlSection += `${
                uniquePrefixes.values().next().value
            }.fun_control=@control_function_${uniquePrefixes.values().next().value};\n`;
            controlSection += `${uniquePrefixes.values().next().value}.fun_event=[];\n`;

            updatedContent += `\n${controlSection}`;
            controlExists = true;
        }

        uniquePrefixes.forEach((prefix) => {
            if (!configExists) {
                updatedContent += `\n% %model configuration\n${prefix}.time.min=0;\n${prefix}.time.max=${timeMax};\n${prefix}.time.id='t[h]';\n${prefix}.time.TAU=${timeTAU};\n${prefix}.time.tspan=${tspan};\n`;
                configExists = true;
            }
        });

        if (!mlmExists) {
            const speciesOptions = extractOptionsFromHmod(updatedContent, "species");
            const controlOptions = extractOptionsFromHmod(updatedContent, "control");
            const parameterOptions = extractOptionsFromHmod(updatedContent, "parameters");
            const compartmentOptions = extractOptionsFromHmod(updatedContent, "compartment");

            setSpeciesOptions(speciesOptions);
            setControlOptions(controlOptions);
            setParameterOptions(parameterOptions);
            setCompartmentOptions(compartmentOptions);

            const mlmOptions = await openMlmModal();
            if (Object.keys(mlmOptions).length > 0) {
                let mlmSection = `% % MLM - Machine Learning Model --------------------------------------------\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.mlm.id = 'mlpnet';\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.mlm.nx = ${
                    mlmOptions.nx
                };\n`;

                mlmOptions.xOptions.forEach((x, index) => {
                    mlmSection += `${uniquePrefixes.values().next().value}.mlm.x(${
                        index + 1
                    }).id = 'anninp${index + 1}';\n`;
                    mlmSection += `${uniquePrefixes.values().next().value}.mlm.x(${
                        index + 1
                    }).val= '${x.val}';\n`;
                    mlmSection += `${uniquePrefixes.values().next().value}.mlm.x(${
                        index + 1
                    }).min= 0;\n`;
                    mlmSection += `${uniquePrefixes.values().next().value}.mlm.x(${
                        index + 1
                    }).max= ${Math.max(...batchData.map((row) => row[x.val]))};\n`;
                });

                mlmSection += `${uniquePrefixes.values().next().value}.mlm.ny = ${
                    mlmOptions.ny
                };\n`;

                mlmOptions.yOptions.forEach((y, index) => {
                    mlmSection += `${uniquePrefixes.values().next().value}.mlm.y(${
                        index + 1
                    }).id = '${y.id}';\n`;
                    mlmSection += `${uniquePrefixes.values().next().value}.mlm.y(${
                        index + 1
                    }).val= 'rann${index + 1}';\n`;
                    mlmSection += `${uniquePrefixes.values().next().value}.mlm.y(${
                        index + 1
                    }).min= 0;\n`;
                    mlmSection += `${uniquePrefixes.values().next().value}.mlm.y(${
                        index + 1
                    }).max= ${Math.max(...batchData.map((row) => row[y.id]))};\n`;
                });

                mlmSection += `${
                    uniquePrefixes.values().next().value
                }.mlm.options={'hidden nodes', [1]};\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.mlm.layer=1;\n`;
                mlmSection += `${
                    uniquePrefixes.values().next().value
                }.mlm.xfun=str2func('autoA_hybmod_anninp');\n`;
                mlmSection += `${
                    uniquePrefixes.values().next().value
                }.mlm.yfun=str2func('autoA_hybmod_rann');\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.symbolic='full-symbolic';\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.symbolic='semi-symbolic';\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.datasource=3;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.datafun=@${
                    uniquePrefixes.values().next().value
                };\n`;

                mlmSection += `%training configuration\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.mode=1;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.method=1;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.jacobian=1;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.hessian=0;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.derivativecheck='off';\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.niter=400;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.niteroptim=1;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.nstep=2;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.display='off';\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.bootstrap=0;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.nensemble=1;\n`;
                mlmSection += `${uniquePrefixes.values().next().value}.crossval=1;\n`;

                updatedContent += `\n${mlmSection}`;
                mlmExists = true;
            }
        }

        updatedContent += `\nend`;

        console.log("Updated Content inside ensureHmodSections: ", updatedContent);
        console.log("Config exists: ", configExists, " Control exists: ", controlExists);
        if (controlExists && configExists && mlmExists) return updatedContent;
    };

    const handleFileChange1 = async (event) => {
        setSelectedFile1(event.target.files[0]);
        setInitialValues(null);
        setHmodOptions({});

        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = async function (e) {
                const content = e.target.result;
                console.log("File content loaded, ensuring HMOD sections...");
                const updatedContent = await ensureHmodSections(
                    content,
                    file2Content,
                    Object.keys(file2Content[0] || {}).filter((key) => key !== "time"),
                    handleOpenHeaderModal,
                    selectedHeaders,
                    mlmOptions
                );

                if (updatedContent) {
                    console.log("Updated Content: ", updatedContent);
                    setFile1Content(updatedContent);

                    const weights = parseWeightsFromHmod(updatedContent);
                    setTrainedWeights(weights);
                } else {
                    console.log("Updated content not set because ensureHmodSections returned null");
                }

                const regexPrefix = /(\w+)\.nspecies=/g;
                const uniquePrefixes = new Set();
                let match;
                while ((match = regexPrefix.exec(updatedContent)) !== null) {
                    uniquePrefixes.add(match[1]);
                }

                let initialValues = {};
                uniquePrefixes.forEach((prefix) => {
                    const hiddenNodesMatch = updatedContent.match(
                        new RegExp(`${prefix}\\.mlm\\.options=\\{'hidden nodes', \\[(.*?)\\]\\};`)
                    );

                    if (hiddenNodesMatch) {
                        console.log("Hidden Nodes Found: ", hiddenNodesMatch[1]);
                    } else {
                        console.log("Hidden Nodes Not Found for prefix: ", prefix);
                    }

                    const hiddenNodes = hiddenNodesMatch
                        ? hiddenNodesMatch[1].trim().replace(/,/g, " ")
                        : "";

                    initialValues = {
                        ...initialValues,
                        hiddenNodes,
                        layer: extractLayerValue(updatedContent, prefix, ""),
                        tau: extractValue(updatedContent, `${prefix}.time.TAU`, ""),
                        mode: extractValue(updatedContent, `${prefix}.mode`, ""),
                        method: extractValue(updatedContent, `${prefix}.method`, ""),
                        jacobian: extractValue(updatedContent, `${prefix}.jacobian`, ""),
                        hessian: extractValue(updatedContent, `${prefix}.hessian`, ""),
                        niter: extractValue(updatedContent, `${prefix}.niter`, ""),
                        nstep: extractValue(updatedContent, `${prefix}.nstep`, ""),
                        bootstrap: extractValue(updatedContent, `${prefix}.bootstrap`, ""),
                    };
                });

                console.log("Initial Values: ", initialValues);
                setInitialValues(initialValues);
                setHmodOptions(initialValues);
            };
            reader.readAsText(file);
        }
        if (progress < 3) setProgress(3);
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
        if (progress < 2) setProgress(2);
    };

    const handleModeChange = (event) => {
        setMode(event.target.value);
        if (event.target.value === "1") {
            if (progress < 4) setProgress(4);
        } else if (event.target.value === "2") {
            if (progress < 5) setProgress(5);
        }
    };

    // Fetch the template HMOD and CSV files from the server
    const getTemplate = (templateType) => {
        fetch("http://localhost:5000/get-template-csv", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ template_type: templateType }),
        })
            .then((response) => response.blob())
            .then((blob) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const fileContent = reader.result;
                    setFile2Content(fileContent);
                };
                reader.readAsText(blob);

                const updatedFileObject = new File([blob], "template.csv", {
                    type: "text/plain",
                });
                setSelectedFile2(updatedFileObject);
                const file = selectedFile2;
                reader.onload = (e) => {
                    const workbook = XLSX.read(e.target.result, { type: "binary" });
                    const sheetName = workbook.SheetNames[0];
                    const worksheet = workbook.Sheets[sheetName];
                    const data = XLSX.utils.sheet_to_json(worksheet, { defval: "" });

                    const separatedBatches = [];
                    let currentBatch = [];

                    data.forEach((row, index) => {
                        const timeValue = row["time"];
                        const nextTimeValue =
                            index + 1 < data.length ? data[index + 1]["time"] : null;

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
                if (progress < 2) setProgress(2);
            })
            .catch((error) => {
                console.error("Error fetching template:", error);
            });

        fetch("http://localhost:5000/get-template-hmod", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ template_type: templateType }),
        })
            .then((response) => response.blob())
            .then((blob) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const fileContent = reader.result;

                    const updatedContent = ensureHmodSections(
                        fileContent,
                        file2Content,
                        Object.keys(file2Content[0] || {}).filter((key) => key !== "time"),
                        handleOpenHeaderModal,
                        selectedHeaders
                    );

                    setFile1Content(updatedContent);
                };
                reader.readAsText(blob);

                const updatedFileObject = new File([blob], "template.hmod", {
                    type: "text/plain",
                });

                setSelectedFile1(updatedFileObject);
                setInitialValues(null);
                setHmodOptions({});
                const file = updatedFileObject;
                if (file) {
                    const reader = new FileReader();
                    reader.onload = async function (e) {
                        const content = e.target.result;

                        const updatedContent = await ensureHmodSections(
                            content,
                            file2Content,
                            Object.keys(file2Content[0] || {}).filter((key) => key !== "time"),
                            handleOpenHeaderModal,
                            selectedHeaders
                        );

                        setFile1Content(updatedContent);

                        const regexPrefix = /(\w+)\.nspecies=/g;
                        const uniquePrefixes = new Set();
                        let match;
                        while ((match = regexPrefix.exec(content)) !== null) {
                            uniquePrefixes.add(match[1]);
                        }

                        let initialValues = {};
                        uniquePrefixes.forEach((prefix) => {
                            const hiddenNodesMatch = content.match(
                                new RegExp(
                                    `${prefix}\\.mlm\\.options=\\{'hidden nodes', \\[(.*?)\\]\\};`
                                )
                            );

                            if (hiddenNodesMatch) {
                                console.log("Hidden Nodes Found: ", hiddenNodesMatch[1]);
                            } else {
                                console.log("Hidden Nodes Not Found for prefix: ", prefix);
                            }

                            const hiddenNodes = hiddenNodesMatch
                                ? hiddenNodesMatch[1].trim().replace(/,/g, " ")
                                : "";

                            initialValues = {
                                ...initialValues,
                                hiddenNodes,
                                layer: extractLayerValue(content, prefix, ""),
                                tau: extractValue(content, `${prefix}.time.TAU`, ""),
                                mode: extractValue(content, `${prefix}.mode`, ""),
                                method: extractValue(content, `${prefix}.method`, ""),
                                jacobian: extractValue(content, `${prefix}.jacobian}`, ""),
                                hessian: extractValue(content, `${prefix}.hessian}`, ""),
                                niter: extractValue(content, `${prefix}.niter}`, ""),
                                nstep: extractValue(content, `${prefix}.nstep}`, ""),
                                bootstrap: extractValue(content, `${prefix}.bootstrap}`, ""),
                            };
                        });

                        console.log("Initial Values: ", initialValues);
                        setHmodOptions(initialValues);
                        setInitialValues(initialValues);
                    };
                    reader.readAsText(file);
                }
                if (progress < 3) setProgress(3);
            })
            .catch((error) => {
                console.error("Error fetching template:", error);
            });
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

        const requiredFields = [
            "hiddenNodes",
            "layer",
            "tau",
            "mode",
            "method",
            "jacobian",
            "hessian",
            "niter",
            "nstep",
            "bootstrap",
        ];
        const missingFields = requiredFields.filter((field) => !hmodOptions[field]);

        if (missingFields.length > 0) {
            alert(
                `The following fields are missing or invalid from the HMOD: ${missingFields.join(
                    ", "
                )}`
            );
            setHmodModalOpen(true);
            return;
        }

        // Ensure HMOD sections before upload
        const updatedContent = await ensureHmodSections(
            file1Content,
            file2Content,
            Object.keys(file2Content[0] || {}).filter((key) => key !== "time"),
            handleOpenHeaderModal,
            selectedHeaders,
            mlmOptions
        );

        if (updatedContent) {
            console.log("asdasdasdasdasasd: ", updatedContent);
            setFile1Content(updatedContent);

            const updatedFile = new Blob([updatedContent], { type: "text/plain" });
            const updatedFileObject = new File([updatedFile], selectedFile1.name, {
                type: selectedFile1.type,
            });
            setFile1Content(updatedContent);
            setSelectedFile1(updatedFileObject);
            uploadFiles(updatedFileObject);
        }
    };

    const uploadFiles = async (updatedFile) => {
        const formData = new FormData();
        formData.append("file1", updatedFile);
        formData.append("file2", selectedFile2);
        formData.append("mode", mode);
        formData.append("userId", auth.currentUser.uid);
        formData.append("description", description);
        formData.append("train_batches", Array.from(train_batches).join(","));
        formData.append("test_batches", Array.from(test_batches).join(","));
        formData.append("user_id", auth.currentUser.uid);
        formData.append("trained_weights", JSON.stringify(trainedWeights));

        console.log("Form Data: ", hmodOptions);
        formData.append("HiddenNodes", hmodOptions.hiddenNodes);
        formData.append("Layer", hmodOptions.layer);
        formData.append("Tau", hmodOptions.tau);
        formData.append("Mode", hmodOptions.mode);
        formData.append("Method", hmodOptions.method);
        formData.append("Jacobian", hmodOptions.jacobian);
        formData.append("Hessian", hmodOptions.hessian);
        formData.append("Niter", hmodOptions.niter);
        formData.append("Nstep", hmodOptions.nstep);
        formData.append("Bootstrap", hmodOptions.bootstrap);
        formData.append("Inputs", JSON.stringify(mlmOptions.xOptions));
        formData.append("Outputs", JSON.stringify(mlmOptions.yOptions));

        formData.append("hiddenOptions", JSON.stringify(hmodOptions));

        console.log("Form Data: ", formData);

        setTrainingModalOpen(true);

        try {
            const response = await fetch("http://localhost:5000/upload", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            setBackendResponse(JSON.stringify(data, null, 2));

            checkRunStatus();
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
                    navigate("/results");
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
                            <h2 style={{ float: "left", marginTop: 0 }}>Simulate Model</h2>
                            {/* 
                            <Button
                                onClick={() => getTemplate(2)}
                                variant='contained'
                                style={{ float: "right", marginTop: 0, margin: 5 }}>
                                Use Template Model 2
                            </Button>
                            <Button
                                onClick={() => getTemplate(1)}
                                variant='contained'
                                style={{ float: "right", marginTop: 0, margin: 5 }}>
                                Use Template Model 1
                            </Button>
                            */}
                        </div>

                        <Grid container spacing={3} columns={20}>
                            <Grid item xs={20} columns={20}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        marginBottom: 0.5,
                                        overflow: "auto",
                                    }}>
                                    <div style={{ display: "flex", alignItems: "center" }}>
                                        <Typography variant='h5'> Step 1: Project Title</Typography>

                                        <Tooltip
                                            title='In this section you can write the Title of the Model you are going to create.'
                                            arrow>
                                            <IconButton size='xsmall' sx={{ ml: 0.5, mb: 1 }}>
                                                <InfoIcon />
                                            </IconButton>
                                        </Tooltip>
                                    </div>
                                </Paper>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        marginBottom: 4,
                                    }}>
                                    <Input
                                        fullWidth
                                        value={description}
                                        onChange={(e) => {
                                            setDescription(e.target.value);
                                            if (e.target.value) {
                                                if (progress < 1) {
                                                    setProgress(1);
                                                }
                                            } else {
                                                setProgress(0);
                                            }
                                        }}
                                    />
                                </Paper>
                            </Grid>

                            <Grid item xs={20}>
                                <Paper sx={{ p: 2, display: "flex", flexDirection: "column" }}>
                                    <Typography variant='h5'>Select a Previous Run</Typography>
                                    <Select
                                        fullWidth
                                        value={selectedRun}
                                        onChange={(e) => handleRunSelection(e.target.value)}
                                        renderValue={(selected) =>
                                            selected ? selected.description : "Select a Run"
                                        }
                                        MenuProps={{
                                            PaperProps: {
                                                style: {
                                                    maxHeight: 48 * 4.5,
                                                },
                                            },
                                        }}>
                                        {completedRuns.map((run) => (
                                            <MenuItem key={run.id} value={run}>
                                                {run.description} -{" "}
                                                {new Date(run.createdAt.toDate()).toLocaleString()}
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </Paper>
                            </Grid>

                            <Grid item xs={20} columns={20}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        marginBottom: 0.5,
                                        overflow: "auto",
                                    }}>
                                    <div style={{ display: "flex", alignItems: "center" }}>
                                        <Typography variant='h5'> Step 2: CSV</Typography>
                                        <Tooltip
                                            title='In this we will ask you to upload the CSV file which is a file containing the information about the batches. After uploading there will be a preview of the file.'
                                            arrow>
                                            <IconButton size='small' sx={{ ml: 1 }}>
                                                <InfoIcon />
                                            </IconButton>
                                        </Tooltip>
                                    </div>
                                </Paper>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        height: 400,
                                        overflow: "auto",
                                    }}>
                                    <p>
                                        {selectedFile2
                                            ? selectedFile2.name
                                            : "Insert your CSV file containing the experimental data here. See structure in the {Link]template file."}
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
                                <div
                                    style={{
                                        display: "flex",
                                        marginTop: "8px",
                                        justifyContent: "flex-end",
                                    }}>
                                    <label htmlFor='csv-upload'>
                                        <Button
                                            fullWidth
                                            variant='contained'
                                            sx={{ height: "60%", marginBottom: 2 }}
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
                                    <DialogTitle>Experimental Data Visualization</DialogTitle>
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
                                        marginBottom: 2,
                                        overflow: "auto",
                                    }}>
                                    <Typography variant='h6'>
                                        Step 3. Please select HMOD hybrid/standard model (.hmod).
                                        The HMOD file is an intermediate format that enables
                                        communication between the essential components of the
                                        mechanistic and hybrid models. You can use the{" "}
                                        <Link href='#'>SBML2HYB</Link> tool to create a hybrid HMOD
                                        model format from any SBML model or see{" "}
                                        <Link href='#'> Example 1</Link> and{" "}
                                        <Link href='#'> Example 2</Link> to edit/create your own
                                        standard HMOD model.
                                    </Typography>
                                </Paper>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        height: 600,
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
                                    <Typography variant='body1' sx={{ mt: 2 }}>
                                        {selectedFile1
                                            ? selectedFile1.name
                                            : "Insert your HMOD file containing the information about the mechanistic model and the settings for the machine learning model here."}
                                    </Typography>
                                    <div style={{ overflowY: "auto", flex: 1 }}>
                                        <pre>{file1Content}</pre>
                                    </div>
                                </Paper>

                                <div
                                    style={{
                                        display: "flex",
                                        marginTop: "8px",
                                        justifyContent: "flex-end",
                                    }}>
                                    <label htmlFor='hmod-upload'>
                                        <Button
                                            component='span'
                                            fullWidth
                                            variant='contained'
                                            sx={{ height: "60%", marginBottom: 2 }}
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

                            <Grid item xs={20}>
                                <Paper sx={{ p: 2, display: "flex", flexDirection: "column" }}>
                                    <div style={{ display: "flex", alignItems: "center" }}>
                                        <Typography variant='h5'>Trained Weights</Typography>
                                        <Tooltip
                                            title='In this section you can upload the trained weights of the model you are going to simulate.'
                                            arrow>
                                            <IconButton size='small' sx={{ ml: 1 }}>
                                                <InfoIcon />
                                            </IconButton>
                                        </Tooltip>
                                    </div>
                                    <Input
                                        fullWidth
                                        value={trainedWeights}
                                        onChange={(e) => {
                                            setTrainedWeights(e.target.value);
                                            if (e.target.value) {
                                                if (progress < 1) {
                                                    setProgress(1);
                                                }
                                            } else {
                                                setProgress(0);
                                            }
                                        }}
                                    />
                                </Paper>
                            </Grid>

                            <Grid item xs={20} columns={20}>
                                <Grid item xs={20}>
                                    <Paper
                                        sx={{
                                            p: 2,
                                            display: "flex",
                                            flexDirection: "column",
                                            overflow: "auto",
                                        }}>
                                        <Typography variant='h6'>
                                            {" "}
                                            Step 4: Please select the batch selection mode.{" "}
                                        </Typography>
                                    </Paper>
                                </Grid>
                                <Grid item xs={10}>
                                    <Paper
                                        sx={{
                                            p: 2,
                                            display: "flex",
                                            flexDirection: "column",
                                            marginBottom: 2,
                                        }}>
                                        <div style={{ display: "flex", alignItems: "center" }}>
                                            <Typography variant='h5'>Batch Selection</Typography>
                                            <Tooltip
                                                title='The experimental dataset selections can be manually adjust or automatically split into training/test sets (2/3 of the data split for the trainning).'
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
                                            sx={{ mt: 2 }}
                                            onChange={handleModeChange}
                                            disabled={progress < 3}>
                                            <MenuItem value={"1"}>Manual</MenuItem>
                                            <MenuItem value={"2"}>Automatic</MenuItem>
                                        </Select>
                                    </Paper>
                                </Grid>

                                {mode === "1" && progress >= 4 && (
                                    <>
                                        <Grid item xs={12}>
                                            <Typography variant='h6'>
                                                Available Batches: {availableBatches.join(", ")}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={6}>
                                            <Typography variant='h6'>
                                                Select Train Batches:{" "}
                                            </Typography>
                                            {availableBatches.map((batch) => (
                                                <div key={batch}>
                                                    <Checkbox
                                                        checked={train_batches.has(batch)}
                                                        onChange={() =>
                                                            handleTrainBatchSelection(batch)
                                                        }
                                                    />
                                                    {batch}
                                                </div>
                                            ))}
                                        </Grid>
                                        <Grid item xs={6}>
                                            <Typography variant='h6'>
                                                Select Test Batches:{" "}
                                            </Typography>
                                            {availableBatches.map((batch) => (
                                                <div key={batch}>
                                                    <Checkbox
                                                        checked={test_batches.has(batch)}
                                                        onChange={() =>
                                                            handleTestBatchSelection(batch)
                                                        }
                                                    />
                                                    {batch}
                                                </div>
                                            ))}
                                        </Grid>
                                        <Grid item xs={12}>
                                            <Button
                                                onClick={handleUpload}
                                                fullWidth
                                                variant='contained'
                                                sx={{ mt: 2 }}
                                                disabled={
                                                    train_batches.size === 0 ||
                                                    test_batches.size === 0 ||
                                                    description === ""
                                                }>
                                                <PublishIcon fontSize='large' />
                                                Start Trainning
                                            </Button>
                                        </Grid>
                                    </>
                                )}

                                {mode === "2" && progress >= 5 && (
                                    <Grid item xs={12}>
                                        <CustomWidthTooltip
                                            title={
                                                tooltipDisplay === "block"
                                                    ? "After clicking on the Upload Information button the information will be uploaded and the run will be created."
                                                    : ""
                                            }
                                            followCursor
                                            arrow>
                                            <Button
                                                onClick={handleUpload}
                                                fullWidth
                                                variant='contained'
                                                sx={{ mt: 2 }}
                                                disabled={description === ""}>
                                                <PublishIcon fontSize='large' />
                                                Start Trainning
                                            </Button>
                                        </CustomWidthTooltip>
                                    </Grid>
                                )}
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
                        <p style={{ margin: 0, textAlign: "center", flex: 1 }}>
                            &copy; {new Date().getFullYear()} Faculdade de Ciências e Tecnologia
                            Universidade NOVA de Lisboa 2024. All rights reserved.
                        </p>

                        <img
                            src='https://www.fct.unl.pt/sites/default/files/images/logo_nova_fct_pt_v.png'
                            width='200px'
                            alt='FCT Logo'
                            style={{ marginLeft: "auto" }}
                        />
                    </footer>
                </Box>
            </Box>
            <HmodModal
                open={hmodModalOpen}
                handleClose={() => setHmodModalOpen(false)}
                handleSave={handleHmodModalSave}
                initialValues={initialValues}
                setHmodOptions={setHmodOptions}
            />
            <TrainingModal open={trainingModalOpen} handleClose={handleCloseTrainingModal} />

            <ControlModalSelection
                open={headerModalOpen}
                headers={headerModalConfig.headers}
                handleClose={headerModalConfig.handleClose}
                onSave={headerModalConfig.onSave}
            />

            <MlmModal
                open={mlmModalOpen.open}
                handleClose={mlmModalOpen.handleClose}
                handleSave={mlmModalOpen.onSave}
                speciesOptions={speciesOptions}
                controlOptions={controlOptions}
                parameterOptions={parameterOptions}
                compartmentOptions={compartmentOptions}
            />
        </ThemeProvider>
    );
}

export default Simulations;
