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
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import { mainListItems, secondaryListItems } from "./ListItems";
import { useNavigate } from "react-router-dom";
import { auth } from "../firebase-config";
import { collection, query, where, getDocs, orderBy, limit } from "firebase/firestore";
import { db } from "../firebase-config";
import { useEffect } from "react";

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

    const navigateToUpload = () => {
        navigate("/");
    };

    const navigateToCreateRun = () => {
        navigate("/upload");
    };

    const [open, setOpen] = React.useState(true);
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
                        height: "99vh",
                        overflow: "auto",
                    }}>
                    <Toolbar />
                    <Container maxWidth='lg' sx={{ mt: 1, mb: 4 }}>
                        <h1>Welcome to Hybpy</h1>
                        <Typography variant='subtitle1' gutterBottom>
                            This tool is design to combine the precision of state-of-the-art machine
                            learning algorithms with the reliability of mechanistic models within a
                            unified structure. This innovative approach offers a user-friendly
                            interface that bridges the gap between complex modeling techniques and
                            practical applications in biosystems engineering.
                        </Typography>
                        <h2>What is Hybrid Modeling?</h2>
                        <Typography variant='subtitle1' gutterBottom>
                            Hybrid modeling is a cutting-edge approach that integrates the
                            predictive power of machine learning algorithms with the foundational
                            principles of mechanistic models. This synergy allows for the creation
                            of models that are not only highly accurate but also deeply insightful,
                            providing a comprehensive understanding of bioprocesses and bio
                            (chemical) systems.
                        </Typography>
                        <h2>The Need for a Python Tool for Hybrid Modeling</h2>
                        <Typography variant='subtitle1' gutterBottom>
                            Despite the proven effectiveness of hybrid models in the process systems
                            engineering field, their adoption has been limited. The primary barrier
                            has been the lack of accessible tools that offer both the sophistication
                            needed for advanced modeling and the simplicity required for widespread
                            use. Our Python tool is the solution to this challenge, offering an
                            open-source, user-friendly platform for analyzing, and simulating hybrid
                            models.
                        </Typography>
                        <h2>How Our Tool Makes a Difference</h2>
                        <Typography variant='subtitle1' gutterBottom>
                            Our tool is designed with the user in mind, simplifying the complex
                            process of hybrid modeling without compromising on power or precision.
                            It enables researchers and practitioners in the biosystems engineering
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
                                    Create Run
                                </Button>{" "}
                                tab to start using the tool. If you have any questions or need
                                assistance, please feel free to contact us.
                            </b>
                        </Typography>
                    </Container>
                </Box>
            </Box>
        </ThemeProvider>
    );
}
