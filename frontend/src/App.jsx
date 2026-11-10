import { useEffect, useState } from "react";

import {
  Activity,
  ArrowUpRight,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  ClipboardList,
  FileText,
  Gauge,
  LayoutDashboard,
  LoaderCircle,
  Menu,
  RefreshCw,
  ShieldCheck,
  Users,
  Wifi,
  Zap,
} from "lucide-react";

import "./App.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || ""
).replace(/\/$/, "");


const INITIAL_CUSTOMER = {
  gender: "Female",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 89.50,
  TotalCharges: 1074.00,
};


const SELECT_FIELDS = [
  {
    section: "Customer profile",
    fields: [
      {
        name: "gender",
        label: "Gender",
        options: ["Female", "Male"],
      },
      {
        name: "SeniorCitizen",
        label: "Senior citizen",
        options: [
          { value: 0, label: "No" },
          { value: 1, label: "Yes" },
        ],
        numeric: true,
      },
      {
        name: "Partner",
        label: "Has partner",
        options: ["Yes", "No"],
      },
      {
        name: "Dependents",
        label: "Has dependents",
        options: ["Yes", "No"],
      },
    ],
  },
  {
    section: "Subscribed services",
    fields: [
      {
        name: "PhoneService",
        label: "Phone service",
        options: ["Yes", "No"],
      },
      {
        name: "MultipleLines",
        label: "Multiple lines",
        options: ["Yes", "No", "No phone service"],
      },
      {
        name: "InternetService",
        label: "Internet service",
        options: ["DSL", "Fiber optic", "No"],
      },
      {
        name: "OnlineSecurity",
        label: "Online security",
        options: ["Yes", "No", "No internet service"],
      },
      {
        name: "OnlineBackup",
        label: "Online backup",
        options: ["Yes", "No", "No internet service"],
      },
      {
        name: "DeviceProtection",
        label: "Device protection",
        options: ["Yes", "No", "No internet service"],
      },
      {
        name: "TechSupport",
        label: "Technical support",
        options: ["Yes", "No", "No internet service"],
      },
      {
        name: "StreamingTV",
        label: "Streaming TV",
        options: ["Yes", "No", "No internet service"],
      },
      {
        name: "StreamingMovies",
        label: "Streaming movies",
        options: ["Yes", "No", "No internet service"],
      },
    ],
  },
  {
    section: "Contract and billing",
    fields: [
      {
        name: "Contract",
        label: "Contract type",
        options: [
          "Month-to-month",
          "One year",
          "Two year",
        ],
      },
      {
        name: "PaperlessBilling",
        label: "Paperless billing",
        options: ["Yes", "No"],
      },
      {
        name: "PaymentMethod",
        label: "Payment method",
        options: [
          "Electronic check",
          "Mailed check",
          "Bank transfer (automatic)",
          "Credit card (automatic)",
        ],
      },
    ],
  },
];


const NUMBER_FIELDS = [
  {
    name: "tenure",
    label: "Customer tenure",
    unit: "Months",
    min: 0,
    max: 72,
    step: 1,
  },
  {
    name: "MonthlyCharges",
    label: "Monthly charges",
    unit: "Currency units",
    min: 0,
    step: 0.01,
  },
  {
    name: "TotalCharges",
    label: "Total charges",
    unit: "Currency units",
    min: 0,
    step: 0.01,
  },
];


function App() {
  const [customer, setCustomer] = useState(INITIAL_CUSTOMER);

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const [apiStatus, setApiStatus] = useState("checking");

  const [sidebarOpen, setSidebarOpen] = useState(false);


  // ----------------------------------------------------------
  // CHECK DJANGO API STATUS
  // ----------------------------------------------------------

  useEffect(() => {
    let active = true;

    async function checkAPI() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/health/`);

        if (!response.ok) {
          throw new Error("API unavailable");
        }

        if (active) {
          setApiStatus("online");
        }

      } catch {
        if (active) {
          setApiStatus("offline");
        }
      }
    }

    checkAPI();

    return () => {
      active = false;
    };
  }, []);


  // ----------------------------------------------------------
  // HANDLE SELECT CHANGES
  // ----------------------------------------------------------

  function handleSelectChange(name, value, numeric = false) {
    setCustomer((previous) => ({
      ...previous,
      [name]: numeric ? Number(value) : value,
    }));
  }


  // ----------------------------------------------------------
  // HANDLE NUMBER INPUT CHANGES
  // ----------------------------------------------------------

  function handleNumberChange(name, value) {
    setCustomer((previous) => ({
      ...previous,
      [name]: Number(value),
    }));
  }


  // ----------------------------------------------------------
  // RESET FORM
  // ----------------------------------------------------------

  function handleReset() {
    setCustomer(INITIAL_CUSTOMER);
    setResult(null);
    setError("");
  }


  // ----------------------------------------------------------
  // SEND CUSTOMER DATA TO DJANGO API
  // ----------------------------------------------------------

  async function handleSubmit(event) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/predict/`, {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify(customer),
      });

      const data = await response.json();

      if (!response.ok) {
        const details = data.errors
          ? JSON.stringify(data.errors)
          : data.detail || "Prediction failed.";

        throw new Error(details);
      }

      setResult(data);

    } catch (err) {
      setError(
        err.message ||
        "Unable to connect to the prediction API."
      );

    } finally {
      setLoading(false);
    }
  }


  // ----------------------------------------------------------
  // FORM FIELD RENDERING
  // ----------------------------------------------------------

  function renderSelect(field) {
    return (
      <div className="field" key={field.name}>
        <label htmlFor={field.name}>
          {field.label}
        </label>

        <div className="select-wrapper">
          <select
            id={field.name}
            value={customer[field.name]}
            onChange={(event) =>
              handleSelectChange(
                field.name,
                event.target.value,
                field.numeric
              )
            }
          >
            {field.options.map((option) => {
              const value =
                typeof option === "object"
                  ? option.value
                  : option;

              const label =
                typeof option === "object"
                  ? option.label
                  : option;

              return (
                <option key={value} value={value}>
                  {label}
                </option>
              );
            })}
          </select>

          <ChevronDown size={15} />
        </div>
      </div>
    );
  }


  function renderNumber(field) {
    return (
      <div className="field" key={field.name}>
        <label htmlFor={field.name}>
          {field.label}
        </label>

        <div className="number-wrapper">
          <input
            id={field.name}
            type="number"
            required
            min={field.min}
            max={field.max}
            step={field.step}
            value={customer[field.name]}
            onChange={(event) =>
              handleNumberChange(
                field.name,
                event.target.value
              )
            }
          />

          <span>{field.unit}</span>
        </div>
      </div>
    );
  }


  return (
    <div className="app-shell">

      {/* ================================================== */}
      {/* SIDEBAR */}
      {/* ================================================== */}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>

        <div className="brand">
          <div className="brand-icon">
            <BrainCircuit size={23} />
          </div>

          <div>
            <h2>ChurnGuard</h2>
            <span>INTELLIGENCE PLATFORM</span>
          </div>

          <button
            className="mobile-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close navigation"
          >
            ×
          </button>
        </div>


        <div className="sidebar-label">
          WORKSPACE
        </div>

        <nav className="side-navigation">

          <a className="nav-link active" href="#overview">
            <LayoutDashboard size={18} />
            <span>Overview</span>
          </a>

          <a className="nav-link" href="#prediction">
            <Activity size={18} />
            <span>Churn prediction</span>
          </a>

          <a className="nav-link" href="#performance">
            <Gauge size={18} />
            <span>Model performance</span>
          </a>

        </nav>


        <div className="sidebar-label second-label">
          RESOURCES
        </div>

        <nav className="side-navigation">

          <a className="nav-link" href="#about">
            <FileText size={18} />
            <span>Project details</span>
          </a>

          <a
            className="nav-link"
            href="https://github.com/"
            target="_blank"
            rel="noreferrer"
          >
            <ArrowUpRight size={18} />
            <span>GitHub</span>
          </a>

        </nav>


        <div className="sidebar-bottom">

          <div className="sidebar-status">
            <span
              className={`status-dot ${
                apiStatus === "online"
                  ? "dot-online"
                  : apiStatus === "offline"
                    ? "dot-offline"
                    : ""
              }`}
            />

            <div>
              <strong>
                {apiStatus === "online"
                  ? "API connected"
                  : apiStatus === "offline"
                    ? "API offline"
                    : "Checking API"}
              </strong>

              <span>Local development server</span>
            </div>
          </div>

          <div className="profile">
            <div className="profile-avatar">
              CG
            </div>

            <div>
              <strong>ChurnGuard</strong>
              <span>ML Dashboard</span>
            </div>
          </div>

        </div>

      </aside>


      {/* ================================================== */}
      {/* MAIN CONTENT */}
      {/* ================================================== */}

      <main className="main-content">

        <header className="topbar">

          <button
            className="menu-button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={22} />
          </button>

          <div className="breadcrumb">
            Workspace <span>/</span> Customer analytics
          </div>

          <div className="topbar-right">

            <div className="api-indicator">
              <span className={`status-dot ${
                apiStatus === "online"
                  ? "dot-online"
                  : apiStatus === "offline"
                    ? "dot-offline"
                    : ""
              }`} />

              {apiStatus === "online"
                ? "System operational"
                : apiStatus === "offline"
                  ? "API unavailable"
                  : "Connecting..."}
            </div>

            <div className="topbar-icon">
              <CircleHelp size={18} />
            </div>

          </div>

        </header>


        <div className="page-container">

          {/* ================================================== */}
          {/* PAGE HEADING */}
          {/* ================================================== */}

          <section className="page-heading" id="overview">

            <div>

              <div className="eyebrow">
                <Zap size={14} />
                MACHINE LEARNING PLATFORM
              </div>

              <h1>Customer intelligence</h1>

              <p>
                Predict customer churn, understand retention
                patterns, and identify customers who may need attention.
              </p>

            </div>

            <div className="dataset-badge">
              <div className="dataset-icon">
                <Users size={19} />
              </div>

              <div>
                <strong>IBM Telco Dataset</strong>
                <span>7,043 customer records</span>
              </div>

            </div>

          </section>


          {/* ================================================== */}
          {/* MODEL PERFORMANCE CARDS */}
          {/* ================================================== */}

          <section className="stats-grid" id="performance">

            <div className="stat-card">

              <div className="stat-top">
                <span>Total records</span>
                <div className="stat-icon purple">
                  <Users size={19} />
                </div>
              </div>

              <h2>7,043</h2>

              <p>Historical customer records</p>

            </div>


            <div className="stat-card">

              <div className="stat-top">
                <span>Recall</span>
                <div className="stat-icon green">
                  <Activity size={19} />
                </div>
              </div>

              <h2>75.67%</h2>

              <p>Held-out test performance</p>

            </div>


            <div className="stat-card">

              <div className="stat-top">
                <span>Precision</span>
                <div className="stat-icon orange">
                  <ShieldCheck size={19} />
                </div>
              </div>

              <h2>52.60%</h2>

              <p>Held-out test performance</p>

            </div>


            <div className="stat-card">

              <div className="stat-top">
                <span>ROC-AUC</span>
                <div className="stat-icon blue">
                  <Gauge size={19} />
                </div>
              </div>

              <h2>84.10%</h2>

              <p>Held-out test performance</p>

            </div>

          </section>


          {/* ================================================== */}
          {/* MAIN DASHBOARD GRID */}
          {/* ================================================== */}

          <section className="dashboard-grid" id="prediction">

            {/* CUSTOMER FORM */}

            <div className="panel prediction-panel">

              <div className="panel-heading">

                <div className="panel-icon">
                  <ClipboardList size={19} />
                </div>

                <div>
                  <h2>Customer information</h2>

                  <p>
                    Enter customer details to generate a prediction.
                  </p>
                </div>

              </div>


              <form onSubmit={handleSubmit}>

                {/* PROFILE */}

                <div className="form-section">

                  <div className="section-heading">
                    <span>01</span>
                    Customer profile
                  </div>

                  <div className="fields-grid">

                    {SELECT_FIELDS[0].fields.map(renderSelect)}

                    {renderNumber(NUMBER_FIELDS[0])}

                  </div>

                </div>


                {/* SERVICES */}

                <div className="form-section">

                  <div className="section-heading">
                    <span>02</span>
                    Subscribed services
                  </div>

                  <div className="fields-grid">

                    {SELECT_FIELDS[1].fields.map(renderSelect)}

                  </div>

                </div>


                {/* BILLING */}

                <div className="form-section">

                  <div className="section-heading">
                    <span>03</span>
                    Contract and billing
                  </div>

                  <div className="fields-grid">

                    {SELECT_FIELDS[2].fields.map(renderSelect)}

                    {NUMBER_FIELDS.slice(1).map(renderNumber)}

                  </div>

                </div>


                {/* ERRORS */}

                {error && (

                  <div className="error-message">
                    <CircleHelp size={17} />
                    <span>{error}</span>
                  </div>

                )}


                {/* ACTIONS */}

                <div className="form-actions">

                  <button
                    type="button"
                    className="reset-button"
                    onClick={handleReset}
                    disabled={loading}
                  >
                    <RefreshCw size={16} />
                    Reset
                  </button>


                  <button
                    type="submit"
                    className="predict-button"
                    disabled={loading || apiStatus === "offline"}
                  >

                    {loading ? (
                      <>
                        <LoaderCircle className="spin" size={18} />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        Generate prediction
                        <ArrowUpRight size={18} />
                      </>
                    )}

                  </button>

                </div>

              </form>

            </div>


            {/* PREDICTION RESULT */}

            <div className="result-column">

              <div className="panel result-panel">

                <div className="panel-heading">

                  <div className="panel-icon result-heading-icon">
                    <BrainCircuit size={19} />
                  </div>

                  <div>
                    <h2>Prediction result</h2>

                    <p>
                      Machine learning analysis
                    </p>
                  </div>

                </div>


                {!result ? (

                  <div className="empty-result">

                    <div className="empty-result-icon">
                      <BrainCircuit size={36} />
                    </div>

                    <h3>Ready for analysis</h3>

                    <p>
                      Enter customer information and generate
                      a prediction to see the model's output.
                    </p>

                    <div className="empty-result-note">
                      <ShieldCheck size={16} />
                      Connected to trained ML pipeline
                    </div>

                  </div>

                ) : (

                  <div className="prediction-result">

                    <div className={
                      result.churn_prediction === 1
                        ? "prediction-status churn"
                        : "prediction-status stay"
                    }>

                      {result.churn_prediction === 1
                        ? "Potential churn detected"
                        : "Customer predicted to stay"}

                    </div>


                    <div className="result-main-icon">

                      {result.churn_prediction === 1 ? (
                        <Activity size={30} />
                      ) : (
                        <CheckCircle2 size={30} />
                      )}

                    </div>


                    <h3>
                      {result.churn_label === "Yes"
                        ? "Churn predicted"
                        : "No churn predicted"}
                    </h3>

                    <p className="result-description">
                      The trained Logistic Regression model
                      classified this customer using the selected
                      decision threshold.
                    </p>


                    <div className="score-card">

                      <div className="score-heading">

                        <span>Churn model score</span>

                        <strong>
                          {Number(result.churn_score).toFixed(4)}
                        </strong>

                      </div>

                      <div className="score-track">

                        <div
                          className={`score-fill ${
                            result.churn_prediction === 1
                              ? "score-risk"
                              : "score-safe"
                          }`}

                          style={{
                            width: `${Math.max(
                              0,
                              Math.min(100, result.churn_score * 100)
                            )}%`
                          }}
                        />

                        <div
                          className="threshold-marker"
                          style={{
                            left: `${result.decision_threshold * 100}%`
                          }}
                        />

                      </div>

                      <div className="score-legend">

                        <span>Model score: 0</span>

                        <span>
                          Threshold: {
                            Number(result.decision_threshold).toFixed(3)
                          }
                        </span>

                        <span>1</span>

                      </div>

                    </div>


                    <div className="result-details">

                      <div>
                        <span>Predicted class</span>
                        <strong>{result.churn_prediction}</strong>
                      </div>

                      <div>
                        <span>Decision threshold</span>
                        <strong>
                          {Number(
                            result.decision_threshold
                          ).toFixed(3)}
                        </strong>
                      </div>

                      <div>
                        <span>Prediction method</span>
                        <strong>Logistic Regression</strong>
                      </div>

                    </div>


                    <div className="score-disclaimer">
                      <CircleHelp size={15} />

                      <span>
                        Model score is uncalibrated and should
                        not be interpreted as a real-world
                        churn probability.
                      </span>

                    </div>

                  </div>

                )}

              </div>


              {/* MODEL INFORMATION */}

              <div className="panel model-card">

                <div className="model-card-title">

                  <div className="model-card-icon">
                    <BrainCircuit size={20} />
                  </div>

                  <div>
                    <h3>Model information</h3>
                    <span>Current prediction engine</span>
                  </div>

                </div>


                <div className="model-info-row">
                  <span>Algorithm</span>
                  <strong>Logistic Regression</strong>
                </div>

                <div className="model-info-row">
                  <span>Class weighting</span>
                  <strong>Balanced</strong>
                </div>

                <div className="model-info-row">
                  <span>Regularization C</span>
                  <strong>0.1</strong>
                </div>

                <div className="model-info-row">
                  <span>Decision threshold</span>
                  <strong>0.547</strong>
                </div>

                <div className="model-ready">
                  <span className="status-dot dot-online" />
                  Model artifact loaded by API
                </div>

              </div>

            </div>

          </section>


          {/* ================================================== */}
          {/* FOOTER */}
          {/* ================================================== */}

          <footer className="dashboard-footer" id="about">

            <div>
              <span className="footer-brand">
                <BrainCircuit size={16} />
                ChurnGuard
              </span>

              <span>
                Customer Churn Prediction · Portfolio ML Project
              </span>
            </div>

            <span>
              Built with React, Django and Scikit-learn
            </span>

          </footer>

        </div>

      </main>

    </div>
  );
}


export default App;
