import * as React from "react";
import { useState } from "react";
import { styled, createTheme, ThemeProvider } from "@mui/material/styles";
import {
    CssBaseline,
    AppBar as MuiAppBar,
    Drawer as MuiDrawer,
    Box,
    Toolbar,
    List,
    Typography,
    Divider,
    IconButton,
    Container,
    RadioGroup,
    FormControlLabel,
    Radio,
    Button,
} from "@mui/material";
import { useEffect } from "react";

import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import { mainListItems, secondaryListItems } from "./ListItems";
import { useNavigate } from "react-router-dom";
import { auth } from "../firebase-config";
import { collection, query, where, getDocs, orderBy, limit } from "firebase/firestore";
import { db } from "../firebase-config";
import logo from "../Image/HYBpyINVIS_logo_BETA.png";
import step1 from "../Image/Guide Print/Step1.png";
import step2 from "../Image/Guide Print/Step2.png";
import step3 from "../Image/Guide Print/Step3.png";
import step4 from "../Image/Guide Print/Step4.png";
import step41 from "../Image/Guide Print/Step4.1.png";
import step42 from "../Image/Guide Print/Step4.2.png";
import step5 from "../Image/Guide Print/Step5.png";
import step51 from "../Image/Guide Print/Step5.1.png";
import step61 from "../Image/Guide Print/Step6.1.png";
import step62 from "../Image/Guide Print/Step6.2.png";
import step7 from "../Image/Guide Print/Step7.png";
import step3a from "../Image/Guide Print/Step3a.png";
import step4b from "../Image/Guide Print/Step4b.png";
import step41b from "../Image/Guide Print/Step4.1b.png";
import step42b from "../Image/Guide Print/Step4.2b.png";

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

export default function LandingPage() {
    const navigate = useNavigate();
    const [open, setOpen] = React.useState(true);
    const [hasFiles, setHasFiles] = useState("");
    const [hasHmod, setHasHmod] = useState("");
    const [hasMlm, setHasMlm] = useState("");

    const navigateToUpload = () => {
        navigate("/");
    };

    const handleFileSelection = (event) => {
        setHasFiles(event.target.value);
    };

    const handleHmodSelection = (event) => {
        setHasHmod(event.target.value);
    };

    const handleMlmSelection = (event) => {
        setHasMlm(event.target.value);
    };

    const navigateToCreateRun = () => {
        navigate("/upload");
    };

    const toggleDrawer = () => {
        setOpen(!open);
    };

    const [runs, setRuns] = React.useState([]);
    const userId = auth.currentUser.uid;

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
                        display: "flex",
                        flexDirection: "column",
                    }}>
                    <Toolbar />
                    <Container maxWidth='lg' sx={{ mt: 1, mb: 4, flex: 1 }}>
                        <h1>Guide</h1>

                        <h2>Step 1: Navigate to "New Project"</h2>
                        <p>
                            To start navigate to "New Project". Here you will be able to start a new
                            training project by uploading your data.
                        </p>
                        <img src={step1} alt='Step 1' width='100%' />

                        <h2>Step 2: Set Title</h2>
                        <p>
                            Once in the "New Project" page, you will be able to set a title for your
                            project.
                        </p>
                        <img src={step2} alt='Step 2' width='100%' />

                        <p>
                            After giving a title you can proceed to upload the files necessary for
                            the trainning.
                        </p>

                        <RadioGroup
                            row
                            aria-label='files'
                            name='files'
                            value={hasFiles}
                            onChange={handleFileSelection}>
                            <FormControlLabel
                                value='ownFiles'
                                control={<Radio />}
                                label='I have my own files'
                            />
                            <FormControlLabel
                                value='exampleFiles'
                                control={<Radio />}
                                label='Use example files from the website'
                            />
                        </RadioGroup>

                        {hasFiles === "ownFiles" ? (
                            <>
                                <h2>Step 3: Upload your data file. </h2>
                                <p>
                                    You can upload you data file by clicking on the "Upload CSV"
                                    button. It needs to be a csv file, if you are not sure of the
                                    format press "Get CSV Template" to download a explanatory file.
                                    After uploading the file there is also a button called "View
                                    Batches" that will show you the data that was uploaded in a
                                    graphical way.
                                </p>
                                <img src={step3} alt='Step 3' width='100%' />

                                <p>
                                    After uploading the data you now need to upload the file
                                    containing the model
                                </p>

                                <RadioGroup
                                    row
                                    aria-label='files'
                                    name='files'
                                    value={hasHmod}
                                    onChange={handleHmodSelection}>
                                    <FormControlLabel
                                        value='ownHmod'
                                        control={<Radio />}
                                        label='I have my own hmod file'
                                    />
                                    <FormControlLabel
                                        value='ownSBML'
                                        control={<Radio />}
                                        label='I have a SBML file'
                                    />
                                </RadioGroup>

                                {hasHmod === "ownHmod" ? (
                                    <>
                                        <h2>Step 4: Upload your hmod file. </h2>

                                        <p>
                                            {" "}
                                            You can upload you hmod file by clicking on the "Upload
                                            HMOD" button.{" "}
                                        </p>

                                        <img src={step4} alt='Step 4' width='100%' />

                                        <p>
                                            After uploading your hmod file, if it does not have a
                                            mlm component a pop up will appear so that you can add
                                            it.
                                        </p>

                                        <RadioGroup
                                            row
                                            aria-label='files'
                                            name='files'
                                            value={hasMlm}
                                            onChange={handleMlmSelection}>
                                            <FormControlLabel
                                                value='ownMlm'
                                                control={<Radio />}
                                                label='My hmod file has a mlm component'
                                            />
                                            <FormControlLabel
                                                value='noMlm'
                                                control={<Radio />}
                                                label='My hmod file does not have a mlm component'
                                            />
                                        </RadioGroup>

                                        {hasMlm === "ownMlm" ? (
                                            <>
                                                <h2>Step 5: Verify and Modify Mlm settings</h2>

                                                <p>
                                                    After uploading the hmod file, you can verify
                                                    the mlm settings and modify them if necessary by
                                                    clicking on the "Edit HMOD Settings" button.
                                                </p>

                                                <img src={step5} alt='Step 5' width='100%' />

                                                <p>
                                                    After clicking on the "Edit HMOD Settings"
                                                    button a pop up will appear with the mlm
                                                    settings and you can modify them as you wish.
                                                    There is also advanced settings that can be
                                                    modified (we recommend to not change them if you
                                                    are not sure about them).
                                                </p>

                                                <img src={step51} alt='Step 5.1' width='100%' />

                                                <h2>Step 6: Select Batch Selection Method</h2>

                                                <p>
                                                    After all the files are uploaded. You can select
                                                    a batch selection method. Ther are two options:
                                                    Random and Manual. Random will select the
                                                    batches randomly and Manual will allow you to
                                                    select the batches you want to use.
                                                </p>

                                                <>
                                                    <img src={step61} alt='Step 6.1' width='100%' />
                                                    <img src={step62} alt='Step 6.2' width='100%' />
                                                </>

                                                <h2>Step 7: Create Run</h2>

                                                <p>
                                                    After all the files are uploaded, the settings
                                                    are set and batches selected, you can create the
                                                    run by clicking on the "Create Training" button.
                                                </p>

                                                <img src={step7} alt='Step 7' width='100%' />
                                            </>
                                        ) : hasMlm === "noMlm" ? (
                                            <>
                                                <h2>Step 4.1: Add Control Variables</h2>

                                                <p>
                                                    The first pop up will apear and you can select
                                                    with variables are going to be used as control
                                                    variables.
                                                </p>

                                                <img src={step41} alt='Step 4.1' width='100%' />

                                                <h2>
                                                    Step 4.2: Set inputs and outputs of the network
                                                </h2>

                                                <p>
                                                    The second pop up will apear and you can select
                                                    the number of inputs and their variables and the
                                                    number of outputs and their variables.
                                                </p>

                                                <img src={step42} alt='Step 4.2' width='100%' />

                                                <h2>Step 5: Verify and Modify Mlm Settings</h2>

                                                <p>
                                                    After adding the mlm component to the hmod file,
                                                    you can verify the mlm settings and modify them
                                                    if necessary by clicking on the "Edit HMOD
                                                    Settings" button.
                                                </p>

                                                <img src={step5} alt='Step 5' width='100%' />

                                                <p>
                                                    After clicking on the "Edit HMOD Settings"
                                                    button a pop up will appear with the mlm
                                                    settings and you can modify them as you wish.
                                                    There is also advanced settings that can be
                                                    modified (we recommend to not change them if you
                                                    are not sure about them).
                                                </p>

                                                <img src={step51} alt='Step 5.1' width='100%' />

                                                <h2>Step 6: Select Batch Selection Method</h2>

                                                <p>
                                                    After all the files are uploaded. You can select
                                                    a batch selection method. Ther are two options:
                                                    Random and Manual. Random will select the
                                                    batches randomly and Manual will allow you to
                                                    select the batches you want to use.
                                                </p>

                                                <>
                                                    <img src={step61} alt='Step 6.1' width='100%' />
                                                    <img src={step62} alt='Step 6.2' width='100%' />
                                                </>

                                                <h2>Step 7: Create Run</h2>

                                                <p>
                                                    After all the files are uploaded, the settings
                                                    are set and batches selected, you can create the
                                                    run by clicking on the "Create Training" button.
                                                </p>

                                                <img src={step7} alt='Step 7' width='100%' />
                                            </>
                                        ) : null}
                                    </>
                                ) : hasHmod === "ownSBML" ? (
                                    <>
                                        <h2>Step 4: Download SBML2HYB</h2>

                                        <p>
                                            If you have a SBML file you can download the SBML2HYB
                                            tool that will convert the SBML file to a hmod file.
                                        </p>

                                        <img src={step4b} alt='Step 4b' width='100%' />
                                        <p>
                                            After downloading the SBML2HYB tool you can convert the
                                            SBML file to a hmod file by running the tool and
                                            clicking the button "Translate SBML File" and select the
                                            SBML file.
                                        </p>

                                        <img src={step41b} alt='Step 4.1b' width='100%' />

                                        <p>
                                            After converting the SBML file to a hmod file you can
                                            save the file and upload it to the website.
                                        </p>

                                        <img src={step42b} alt='Step 4.2b' width='100%' />

                                        <h2>Step 5: Upload your hmod file. </h2>

                                        <p>
                                            {" "}
                                            You can upload you hmod file by clicking on the "Upload
                                            HMOD" button.{" "}
                                        </p>

                                        <img src={step4} alt='Step 4' width='100%' />

                                        <p>
                                            After uploading your hmod file, if it does not have a
                                            mlm component a pop up will appear so that you can add
                                            it.
                                        </p>

                                        <h2>Step 5.1: Add Control Variables</h2>

                                        <p>
                                            The first pop up will apear and you can select with
                                            variables are going to be used as control variables.
                                        </p>

                                        <img src={step41} alt='Step 4.1' width='100%' />

                                        <h2>Step 5.2: Set inputs and outputs of the network</h2>

                                        <p>
                                            The second pop up will apear and you can select the
                                            number of inputs and their variables and the number of
                                            outputs and their variables.
                                        </p>

                                        <img src={step42} alt='Step 4.2' width='100%' />

                                        <h2>Step 6: Verify and Modify Mlm Settings</h2>

                                        <p>
                                            After adding the mlm component to the hmod file, you can
                                            verify the mlm settings and modify them if necessary by
                                            clicking on the "Edit HMOD Settings" button.
                                        </p>

                                        <img src={step5} alt='Step 5' width='100%' />

                                        <p>
                                            After clicking on the "Edit HMOD Settings" button a pop
                                            up will appear with the mlm settings and you can modify
                                            them as you wish. There is also advanced settings that
                                            can be modified (we recommend to not change them if you
                                            are not sure about them).
                                        </p>

                                        <img src={step51} alt='Step 5.1' width='100%' />

                                        <h2>Step 7: Select Batch Selection Method</h2>

                                        <p>
                                            After all the files are uploaded. You can select a batch
                                            selection method. Ther are two options: Random and
                                            Manual. Random will select the batches randomly and
                                            Manual will allow you to select the batches you want to
                                            use.
                                        </p>

                                        <>
                                            <img src={step61} alt='Step 6.1' width='100%' />
                                            <img src={step62} alt='Step 6.2' width='100%' />
                                        </>

                                        <h2>Step 8: Create Run</h2>

                                        <p>
                                            After all the files are uploaded, the settings are set
                                            and batches selected, you can create the run by clicking
                                            on the "Create Training" button.
                                        </p>

                                        <img src={step7} alt='Step 7' width='100%' />
                                    </>
                                ) : null}
                            </>
                        ) : hasFiles === "exampleFiles" ? (
                            <>
                                <h2>Step 3: Select Example </h2>

                                <p>
                                    If you don't have your own files you can use the example files
                                    provided on the website. There are 2 examples available.
                                </p>

                                <img src={step3a} alt='Step 3a' width='100%' />

                                <h2>Step 4: Verify Data</h2>

                                <p>
                                    After selecting the example files you can verify the data by
                                    clicking on the "View Batches" button that will show you the
                                    data that was uploaded in a graphical way.
                                </p>

                                <img src={step3a} alt='Step 3a' width='100%' />

                                <h2>Step 5: Verify and Modify Mlm settings</h2>

                                <p>
                                    You can verify the mlm settings and modify them if necessary by
                                    clicking on the "Edit HMOD Settings" button.
                                </p>

                                <img src={step5} alt='Step 5' width='100%' />

                                <p>
                                    After clicking on the "Edit HMOD Settings" button a pop up will
                                    appear with the mlm settings and you can modify them as you
                                    wish. There is also advanced settings that can be modified (we
                                    recommend to not change them if you are not sure about them).
                                </p>

                                <img src={step51} alt='Step 5.1' width='100%' />

                                <h2>Step 6: Select Batch Selection Method</h2>

                                <p>
                                    After all the files are uploaded. You can select a batch
                                    selection method. Ther are two options: Random and Manual.
                                    Random will select the batches randomly and Manual will allow
                                    you to select the batches you want to use.
                                </p>

                                <>
                                    <img src={step61} alt='Step 6.1' width='100%' />
                                    <img src={step62} alt='Step 6.2' width='100%' />
                                </>

                                <h2>Step 7: Create Run</h2>

                                <p>
                                    After all the files are uploaded, the settings are set and
                                    batches selected, you can create the run by clicking on the
                                    "Create Training" button.
                                </p>

                                <img src={step7} alt='Step 7' width='100%' />
                            </>
                        ) : null}
                    </Container>
                </Box>
            </Box>
        </ThemeProvider>
    );
}
