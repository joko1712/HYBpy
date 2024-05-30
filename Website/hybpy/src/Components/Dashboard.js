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
import { collection, query, where, getDocs, orderBy, limit } from "firebase/firestore";
import { db } from "../firebase-config";
import { useEffect } from "react";
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

const defaultTheme = createTheme();

export default function Dashboard() {
    const navigate = useNavigate();

    const navigateToUpload = () => {
        navigate("/");
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
                            <IconButton
                                edge='start'
                                color='inherit'
                                size='small'
                                onClick={() => navigateToUpload()}>
                                <img src={logo} alt='logo' width='200' height='75' />
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
                        height: "98vh",
                        overflow: "auto",
                    }}>
                    <Toolbar />
                    <Container maxWidth='lg' sx={{ mt: 4, mb: 4 }}>
                        <Grid container spacing={3}>
                            {/* Recent Runs */}
                            <Grid item xs={12} md={8} lg={9}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        display: "flex",
                                        flexDirection: "column",
                                        height: 240,
                                    }}>
                                    {runs.length > 0 ? (
                                        <ImageList cols={3} gap={8}>
                                            {runs[0].plot_urls.map((url) => (
                                                <ImageListItem key={url}>
                                                    <img src={url} alt='Run Plot' loading='lazy' />
                                                </ImageListItem>
                                            ))}
                                        </ImageList>
                                    ) : (
                                        <Typography>No plots available</Typography>
                                    )}
                                </Paper>
                            </Grid>
                            {/* Recent Run Details */}
                            <Grid item xs={12} md={4} lg={3}>
                                <Paper sx={{ p: 2, display: "flex", flexDirection: "column" }}>
                                    {runs.length > 0 ? (
                                        <>
                                            <Typography variant='h6' gutterBottom>
                                                Run Details
                                            </Typography>
                                            <Typography>{`HMOD: ${runs[0].file1_name}`}</Typography>
                                            <Typography>{`CSV: ${runs[0].file2_name}`}</Typography>
                                            <Typography>{`Mode: ${runs[0].mode}`}</Typography>
                                            <Typography>{`Description: ${runs[0].description}`}</Typography>
                                        </>
                                    ) : (
                                        <Typography>No recent run details</Typography>
                                    )}
                                </Paper>
                            </Grid>
                        </Grid>
                    </Container>
                </Box>
            </Box>
        </ThemeProvider>
    );
}
