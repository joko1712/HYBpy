import React, { useState } from "react";
import Avatar from "@mui/material/Avatar";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import Grid from "@mui/material/Grid";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import { styled, createTheme, ThemeProvider } from "@mui/material/styles";
import { auth, signInWithGoogle } from "../firebase-config";
import { signInWithEmailAndPassword } from "firebase/auth";
import GoogleIcon from "@mui/icons-material/Google";
import { useNavigate } from "react-router-dom";
import {
    AppBar as MuiAppBar,
    Box,
    Toolbar,
    Typography,
    IconButton,
    Container,
    Link,
    Button,
} from "@mui/material";
import logo from "../Image/HYBpyINVIS_logo_BETA.png";

const defaultTheme = createTheme();

const AppBar = styled(MuiAppBar)(({ theme }) => ({
    zIndex: theme.zIndex.drawer + 1,
}));

function Login({ manualSetCurrentUser }) {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const navigate = useNavigate();

    const loginUser = async (e) => {
        e.preventDefault();
        setError("");

        try {
            const userCredential = await signInWithEmailAndPassword(auth, email, password);
            if (userCredential.user.emailVerified) {
                console.log("Email is verified");
                manualSetCurrentUser(userCredential.user);
                navigate("/");
            } else {
                console.log("Email is not verified");
                window.alert("Please verify your email address.");
            }
        } catch (error) {
            console.error("Login error:", error);
            setError(error.message);
            window.alert("Invalid email or password.");
        }

        setEmail("");
        setPassword("");
    };

    const loginWithGoogle = () => {
        setError("");

        signInWithGoogle(auth)
            .then((res) => {
                console.log(res.user);
            })
            .catch((err) => setError(err.message));
    };

    return (
        <ThemeProvider theme={defaultTheme}>
            <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100%" }}>
                <CssBaseline />
                <AppBar position="fixed" color="primary">
                    <Toolbar sx={{ pr: "2px" }}>
                        <IconButton edge="start" color="inherit" size="small">
                            <img src={logo} alt="logo" width="200" height="75" />
                        </IconButton>
                    </Toolbar>
                </AppBar>

                <Container component="main" maxWidth="xs" sx={{ display: "flex", flexDirection: "column", alignItems: "center", mt: 10, flexGrow: 1, width: "100%" }}>
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", mt: 2 }}>
                        <Avatar sx={{ m: 1, bgcolor: "secondary.main" }}>
                            <LockOutlinedIcon />
                        </Avatar>
                        <Typography component="h1" variant="h5">
                            Sign In
                        </Typography>
                        <Box component="form" onSubmit={loginUser} noValidate sx={{ mt: 1 }}>
                            <TextField
                                margin="normal"
                                required
                                fullWidth
                                id="email"
                                label="Email Address"
                                name="email"
                                autoComplete="email"
                                autoFocus
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                            />
                            <TextField
                                margin="normal"
                                required
                                fullWidth
                                name="password"
                                label="Password"
                                type="password"
                                id="password"
                                autoComplete="current-password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                            <Button type="submit" fullWidth variant="contained" sx={{ mt: 3, mb: 2 }}>
                                Sign In
                            </Button>
                            <Button
                                startIcon={<GoogleIcon />}
                                onClick={loginWithGoogle}
                                fullWidth
                                variant="contained"
                                sx={{ mb: 2 }}
                            >
                                Sign in with Google
                            </Button>
                            {error && <Typography color="error" sx={{ mt: 1 }}>{error}</Typography>}
                            <Grid container justifyContent="flex-end">
                                <Grid item>
                                    <Link href="/register" variant="body2">
                                        {"Don't have an account? Sign up"}
                                    </Link>
                                </Grid>
                            </Grid>
                        </Box>
                    </Box>
                </Container>

                <Box
                    component="footer"
                    sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "1em",
                        backgroundColor: "#f1f1f1",
                        width: "100%",
                        position: "fixed",
                        bottom: 0,
                        left: 0,
                    }}
                >
                    <Typography variant="body2" align="center" sx={{ flexGrow: 1 }}>
                        &copy; {new Date().getFullYear()} Faculdade de Ciências e Tecnologia Universidade NOVA de Lisboa. All rights reserved.
                    </Typography>
                    <img
                        src="https://www.fct.unl.pt/sites/default/files/images/logo_nova_fct_pt_v.png"
                        width="75"
                        alt="FCT Logo"
                        style={{ marginLeft: "auto" }}
                    />
                </Box>
            </Box>
        </ThemeProvider>
    );
}
export default Login;