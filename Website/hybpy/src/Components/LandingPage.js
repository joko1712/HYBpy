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
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import MenuIcon from "@mui/icons-material/Menu";
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
import { useEffect } from "react";
import logo from "../Image/HYBpyINVIS_logo.png";
import hybrid from "../Image/hybridmodel.jpg";
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
        textAlign: "center",
    },
}));

const defaultTheme = createTheme();

export default function LandingPage() {

    const navigateToUpload = () => {
        navigate("/");
    };

    const navigateToCreateRun = () => {
        navigate("/upload");
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
                    <Container maxWidth='lg' sx={{ mt: 1, mb: 4 }}>
                        <h1>Welcome to HYBpy</h1>
                        <Divider
                            sx={{
                                my: 2,
                                borderBottomWidth: 2,
                            }}
                        />
                        <Typography variant='subtitle1' gutterBottom>
                            HYBpy integrates deep learning algorithms with the
                            robustness of mechanistic modelling into a unified
                            hybrid modelling framework for bioprocess
                            engineering. It streamlines the development and
                            analysis of hybrid models for biological systems,
                            providing an intuitive and user-friendly interface
                            that bridges the gap between complex mechanistic
                            models and machine learning.
                        </Typography>
                        <h2>What is Hybrid Modeling?</h2>
                        <Divider
                            sx={{
                                my: 1,
                                borderBottomWidth: 1,
                            }}
                        />
                        <Typography variant='subtitle1' gutterBottom>
                            Hybrid modelling can be defined as the integration
                            of mechanistic (parametric) models with machine
                            learning methods—such as artificial neural networks
                            (nonparametric models)—within a unified mathematical
                            framework. This combination leverages the strengths
                            of both approaches, enhancing prediction accuracy
                            while reducing the amount of experimental data
                            required for process modelling. HYBpy implements a
                            generic hybrid modelling framework for biological
                            systems that integrates deep neural networks with
                            prior knowledge expressed through ordinary
                            differential equations (ODEs) (Pinto et al., 2022).
                            The ODEs system is typically derived from
                            compartment material balance equations encoded in
                            Systems Biology Markup Language (SBML).
                        </Typography>
                        <div
                            style={{
                                display: "flex",
                                justifyContent: "center",
                                alignItems: "center",
                            }}>
                            <img
                                src={hybrid}
                                alt='hybrid model'
                                width='600'
                                height='270'
                            />
                        </div>
                        <div
                            style={{
                                display: "flex",
                                justifyContent: "center",
                                alignItems: "center",
                                marginTop: "10px",
                            }}>
                            <Typography variant='caption'>
                                General deep hybrid model structure for
                                biological systems José Pinto, Mykaella Mestre,
                                J. Ramos, Rafael S. Costa, Gerald Striedner, Rui
                                Oliveira, A general deep hybrid model for
                                bioreactor systems: Combining first principles
                                with deep neural networks, Computers & Chemical
                                Engineering, Volume 165, 2022, 107952,
                                <Link href='https://doi.org/10.1016/j.compchemeng.2022.107952'>
                                    https://doi.org/10.1016/j.compchemeng.2022.107952
                                </Link>
                                .
                            </Typography>
                        </div>
                        <h2>Why HYBpy?</h2>
                        <Divider
                            sx={{
                                my: 1,
                                borderBottomWidth: 1,
                            }}
                        />
                        <Typography variant='subtitle1' gutterBottom>
                            HYBpy streamlines the development of complex hybrid
                            models for biological systems. Mechanistic models
                            encoded in SBML can be seamlessly integrated with
                            deep neural networks, creating powerful hybrid
                            ensembles that can be trained using state-of-the-art
                            deep learning algorithms. The resulting hybrid
                            models are also encoded in SBML and can be deployed
                            in public databases. By simplifying the
                            implementation of hybrid modeling approaches for
                            biological system modeling and optimization, HYBpy
                            promotes the adoption of AI-driven methodologies in
                            bioprocess engineering.
                        </Typography>
                        <h2>Features</h2>
                        <Divider
                            sx={{
                                my: 1,
                                borderBottomWidth: 1,
                            }}
                        />
                        <Typography variant='subtitle1' gutterBottom>
                            Our tool is designed with the user in mind,
                            simplifying the complex process of hybrid modeling
                            without compromising on power or precision. It
                            enables researchers and practitioners in the
                            bioprocesses engineering community to:
                            <ul>
                                <li>
                                    <b>Construct Hybrid Models:</b> Easily
                                    integrate machine learning algorithms with
                                    mechanistic models to address complex
                                    modeling challenges.
                                </li>
                                <li>
                                    <b>Analyze and Simulate:</b> Perform
                                    detailed analyses and simulations to
                                    understand and predict the behavior of
                                    bioprocesses and bio (chemical) systems.
                                </li>
                                <li>
                                    <b>Accelerate Research and Development:</b>{" "}
                                    Reduce the time and resources required to
                                    develop and test computational models,
                                    speeding up innovation.
                                </li>
                            </ul>
                        </Typography>
                        <Divider
                            sx={{
                                my: 1,
                                borderBottomWidth: 1,
                            }}
                        />
                        <div
                            style={{
                                display: "flex",
                                justifyContent: "center",
                                alignItems: "center",
                            }}>
                            <Typography
                                align='center'
                                variant='h6'
                                gutterBottom
                                sx={{ maxWidth: "70%" }}>
                                <b>
                                    Just proceed to the{" "}
                                    <Button
                                        color='inherit'
                                        variant='text'
                                        onClick={() => navigateToCreateRun()}>
                                        New Project
                                    </Button>{" "}
                                    tab to start using the tool. If you have any
                                    questions or need assistance, please feel
                                    free to{" "}
                                    <Button
                                        color='inherit'
                                        variant='text'
                                        onClick={handleContactUsClick}>
                                        contact us
                                    </Button>
                                    .
                                </b>
                            </Typography>
                        </div>
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
