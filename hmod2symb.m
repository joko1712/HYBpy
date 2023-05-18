function [t, State, fState, rates, anninp, rann, ucontrol,w]=hmod2symb(projhyb)

        %Create the symbolic variables
        totalsyms=["t","dummyarg1","dummyarg2","w"]; %fixed variables, t, dummyargs and w
        for n=1:projhyb.nspecies
            totalsyms=[totalsyms,projhyb.species(n).id]; %Species
        end
        for n=1:projhyb.ncompartments
            totalsyms=[totalsyms,projhyb.compartment(n).id]; %Compartments
        end
        for n=1:projhyb.nparameters
            totalsyms=[totalsyms,projhyb.parameters(n).id]; %Parameters
        end
        for n=1:projhyb.nruleAss
            totalsyms=[totalsyms,projhyb.ruleAss(n).id]; %Assigned parameters
        end
        for n=1:projhyb.nreaction
            totalsyms=[totalsyms,projhyb.reaction(n).id]; %Reactions
        end
        for n=1:projhyb.ncontrol
            totalsyms=[totalsyms,projhyb.control(n).id]; %Control variables
        end
        anninp=[];
        rann=[];
        for n=1:projhyb.mlm.nx
            totalsyms=[totalsyms,projhyb.mlm.x(n).id]; %Ann inputs
            anninp=[anninp;str2sym(projhyb.mlm.x(n).val)/projhyb.mlm.x(n).max]; %vector of Ann inputs
        end
        for n=1:projhyb.mlm.ny
            totalsyms=[totalsyms,projhyb.mlm.y(n).id]; %Ann outputs
            totalsyms=[totalsyms,projhyb.mlm.y(n).val];
            rann=[rann;str2sym(projhyb.mlm.y(n).val)];
        end
        syms(totalsyms)
        %-----------
        %Species data
        nspecies=projhyb.nspecies; %number of species
        Species=[]; %initialize species vector
        
        for n=1:projhyb.nspecies
            Species=[Species;str2sym(projhyb.species(n).id)]; %add the symbolic specie to a vector
            projhyb.species(n).dcomp=0; %Initialize the associated compartment rate rule (dV/dt=0 if constant compartment)
            for m=1:projhyb.nraterules
                if strcmp(projhyb.raterules(m).id,projhyb.species(n).compartment) %Check if there is a rate rule for the compartment
                    projhyb.species(n).dcomp=str2sym(projhyb.raterules(m).val); %Associate the rate rule of the compartment to the specie
                end
            end
        end
        %---------
        %Compartment data
        ncompartments=projhyb.ncompartments; %number of compartments
        Compartments=[]; %Initialize compartment vector
        
        for n=1:projhyb.ncompartments
            Compartments=[Compartments;str2sym(projhyb.compartment(n).id)]; %Add compartment to the list
        end
        %---------
        %Create neural network
        for n=1:length(projhyb.mlm.x)
            fun=strcat(projhyb.mlm.x(n).id,'=',projhyb.mlm.x(n).val); %Check each of the network inputs
            eval(fun)
        end
        no=projhyb.mlm.ny; %Check number of outputs
        for n=1:length(projhyb.mlm.y)
            fun=strcat(projhyb.mlm.y(n).id,'=',projhyb.mlm.y(n).val); %Check each of the network outputs
            eval(fun)
        end
        %---------
        %Check for global parameter assignments
        for n=1:projhyb.nparameters
            if strcmp(projhyb.parameters(n).reaction,'global')
                fun=strcat(projhyb.parameters(n).id,'=',projhyb.parameters(n).val); %Assign the global parameters
                eval(fun);
            end
        end
        %--------
        %Assignments rules
        for n=1:projhyb.nruleAss
            fun=strcat(projhyb.ruleAss(n).id,'=',projhyb.ruleAss(n).val);
            eval(fun);
        end
        %--------
        %Rate rule assignments
        nraterules = projhyb.nraterules; %Check number of rule assignments
        Raterules=[]; %Initialize rate variables
        fRaterules=[]; %Initialize rate rules
        for n=1:projhyb.nraterules
            Raterules=[Raterules;str2sym(projhyb.raterules(n).id)]; %Add variable
            fRaterules=[fRaterules;str2sym(projhyb.raterules(n).val)];%Add rate rule of the variable
        end
        %--------
        %Control assignments
        ncontrol=projhyb.ncontrol; %Check number of control
        ucontrol=[]; %Initialzie control variables
        for n=1:projhyb.ncontrol
            ucontrol=[ucontrol;str2sym(projhyb.control(n).id)]; %add the symbolic control to a vector
        end
        %-------
        %Reaction assignments
        stoichm=[]; %Initialize a stoichiometric matrix
        
        for n=1:projhyb.nreaction
            for m=1:projhyb.nparameters
                if strcmp(projhyb.parameters(m).reaction,projhyb.reaction(n).id)
                    fun=strcat(projhyb.parameters(m).id,'=',projhyb.parameters(m).val); %Create the vector of reactions
                    eval(fun);
                end 
            end
            reaction(n,1)=eval(projhyb.reaction(n).rate);
            nval=str2sym(projhyb.reaction(n).Y);
            stoichm=[stoichm;eval(nval)];
        end
        
        ratefuns=[];
        
        for n=1:projhyb.nraterules
            ratefuns = [ratefuns;eval(projhyb.raterules(n).val)];
            ratevars(n)= str2sym(projhyb.raterules(n).id);
        end
        
        for n=1:projhyb.nspecies
            rates(n,1)=sum(reaction.*stoichm(:,n));
        end
        
        for n=1:projhyb.nspecies
            fSpecies(n,1)=rates(n,1)-(projhyb.species(n).dcomp/str2sym(projhyb.species(n).compartment)*str2sym(projhyb.species(n).id)); 
        end
        %-------
        %Create final outputs
        State = [Species; Raterules]; 
        fState = [fSpecies; fRaterules];
        fState=subs(fState);
        nstate= nspecies + nraterules;
        w= []; %in the semisymvolic case 
        %-------
    end