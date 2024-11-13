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
import { collection, query, where, getDocs, orderBy, limit } from "firebase/firestore";
import { db } from "../firebase-config";
import { useEffect } from "react";
import logo from "../Image/HYBpyINVIS_logo_BETA.png";
import hybrid from "../Image/hybridmodel.jpg";
import { Link } from "@mui/material";

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
    useEffect(() => {
        const email1 = process.env.EMAIL1;
        const email2 = process.env.EMAIL2;

        const email1Element = document.getElementById("email1");
        const email2Element = document.getElementById("email2");

        if (email1Element) {
            email1Element.innerHTML = `<a href="mailto:${email1}">José Pedreira</a>`;
        }

        if (email2Element) {
            email2Element.innerHTML = `<a href="mailto:${email2}">Rafael Costa</a>`;
        }
    }, []);

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
    const [open, setOpen] = React.useState(localStorage.getItem("drawerOpen") === "true");
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
                        <Typography variant='subtitle1' gutterBottom>
                            This tool is design to combine state-of-the-art machine learning
                            algorithms with the reliability of mechanistic models within a unified
                            structure and to simplify the construction and analyses of a hybrid
                            model. This innovative approach offers a user-friendly interface that
                            bridges the gap between complex hybrid modeling techniques and practical
                            applications in bioprocesses engineering.
                        </Typography>
                        <h2>What is Hybrid Modeling?</h2>
                        <Typography variant='subtitle1' gutterBottom>
                            Hybrid modeling is a cutting-edge approach that integrates the
                            predictive power of machine learning algorithms (Nonparametric model)
                            with the foundational principles of mechanistic models (Parametric
                            model). This synergy allows for the creation of models that are not only
                            highly accurate but also deeply insightful, providing a comprehensive
                            understanding of bioprocesses and bio (chemical) systems.
                        </Typography>
                        <div
                            style={{
                                display: "flex",
                                justifyContent: "center",
                                alignItems: "center",
                            }}>
                            <img src={hybrid} alt='hybrid model' width='600' height='270' />
                        </div>
                        <div
                            style={{
                                display: "flex",
                                justifyContent: "center",
                                alignItems: "center",
                                marginTop: "10px",
                            }}>
                            <Typography variant='caption'>
                                Example of a typical structure for a hybrid model.
                            </Typography>
                        </div>
                        <h2>Why HYBpy?</h2>
                        <Typography variant='subtitle1' gutterBottom>
                            Despite the proven effectiveness of hybrid models in the process systems
                            engineering field, their adoption has been limited. The primary barrier
                            has been the lack of accessible tools that offer both the sophistication
                            needed for advanced modeling and the simplicity required for widespread
                            use. Our Python tool is the solution to this challenge, offering an
                            open-source, user-friendly platform for analyzing, and simulating hybrid
                            models.
                        </Typography>
                        <h2>Features</h2>
                        <Typography variant='subtitle1' gutterBottom>
                            Our tool is designed with the user in mind, simplifying the complex
                            process of hybrid modeling without compromising on power or precision.
                            It enables researchers and practitioners in the bioprocesses engineering
                            community to:
                            <ul>
                                <li>
                                    <b>Construct Hybrid Models:</b> Easily integrate machine
                                    learning algorithms with mechanistic models to address complex
                                    modeling challenges.
                                </li>
                                <li>
                                    <b>Analyze and Simulate:</b> Perform detailed analyses and
                                    simulations to understand and predict the behavior of
                                    bioprocesses and bio (chemical) systems.
                                </li>
                                <li>
                                    <b>Accelerate Research and Development:</b> Reduce the time and
                                    resources required to develop and test computational models,
                                    speeding up innovation.
                                </li>
                            </ul>
                        </Typography>
                        <Typography variant='h6' gutterBottom>
                            <b>
                                Just proceed to the{" "}
                                <Button
                                    color='inherit'
                                    variant='text'
                                    onClick={() => navigateToCreateRun()}>
                                    New Project
                                </Button>{" "}
                                tab to start using the tool. If you have any questions or need
                                assistance, please feel free to contact us contact us at:{" "}
                                <span id='email1'></span> or <span id='email2'></span>.
                            </b>
                        </Typography>
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
                            width='75px'
                            alt='FCT Logo'
                            style={{ marginLeft: "auto" }}
                        />
                    </footer>
                </Box>
            </Box>
        </ThemeProvider>
    );
}
