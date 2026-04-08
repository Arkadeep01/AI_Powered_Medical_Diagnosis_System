import { useEffect, useMemo, useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

const NORMAL_RANGE_DETAILS = {
  'diabetes': 'Normal values: Glucose 70-99 mg/dL, Blood pressure 80-120 mmHg, BMI 18.5-24.9, Insulin 2-25 µU/mL, Diabetes pedigree function 0.0-0.5.',
  'heart-disease': 'Normal values: Resting blood pressure 90-120 mmHg, Cholesterol under 200 mg/dL, Max heart rate 100-200 bpm, Chest pain type 0-3.',
  'fetal-health': 'Normal values: Baseline 110-160, Accelerations 2 or more, Short term variability 3-5, Long term variability stable.',
  'lung-disease': 'Normal values: Non-smoker, no yellow fingers, no chronic disease, no coughing, no shortness of breath.',
  'parkinson': 'Normal values: Typical voice measures fall within expected adult speech range and low jitter/shimmer.',
  'hypothyroid': 'Normal values: TSH 0.4-4.0 µIU/mL, T3 80-180 ng/dL, TT4 5-12 µg/dL, no thyroid surgery or medication.',
  'migraine': 'Normal values: Headache frequency 0-4/month, duration under 4 hours, no nausea, no photophobia, no sensory changes.',
  'preprocessed-hypothyroid': 'Normal values: TSH 0.4-4.0 µIU/mL, T3 measured normal, no thyroid medication or surgery.',
  'preprocessed-lungs-disease': 'Normal values: Non-smoker, no yellow fingers, no chronic disease, no chest pain, no cough, no breathlessness.',
}

const getNormalRangeDetails = (id) => NORMAL_RANGE_DETAILS[id] || 'Enter your values and compare them with the expected healthy ranges for this diagnosis.'

const normalizeYesNoLabel = (label) => {
  if (!label) {
    return ''
  }

  const normalized = label.toLowerCase()
  const positiveKeywords = ['positive', 'diabetic', 'disease', 'pathological', 'suspect', 'yes', 'cancer']
  const negativeKeywords = ['negative', 'non-', 'no ', 'healthy', 'normal']

  if (positiveKeywords.some((keyword) => normalized.includes(keyword))) {
    return 'Yes'
  }
  if (negativeKeywords.some((keyword) => normalized.includes(keyword))) {
    return 'No'
  }
  return label
}

function App() {
  const [diseases, setDiseases] = useState([])
  const [selectedId, setSelectedId] = useState('')
  const [formState, setFormState] = useState({})
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [metadataError, setMetadataError] = useState(null)

  useEffect(() => {
    const loadMetadata = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/metadata`)
        if (!response.ok) {
          throw new Error('Unable to load model metadata')
        }
        const data = await response.json()
        const modelList = Object.entries(data).map(([id, model]) => ({
          id,
          name: model.name,
          fields: model.fields,
          description: model.description || getNormalRangeDetails(id),
        }))
        setDiseases(modelList)
        if (modelList.length > 0) {
          setSelectedId(modelList[0].id)
        }
      } catch (err) {
        setMetadataError(err.message)
      }
    }

    loadMetadata()
  }, [])

  const selectedDisease = useMemo(
    () => diseases.find((item) => item.id === selectedId),
    [diseases, selectedId]
  )

  useEffect(() => {
    if (!selectedDisease) {
      return
    }

    const initialState = selectedDisease.fields.reduce((acc, field) => {
      if (field.type === 'select') {
        acc[field.name] = field.options[0]?.value ?? ''
      } else {
        acc[field.name] = ''
      }
      return acc
    }, {})

    setFormState(initialState)
    setResult(null)
    setError(null)
  }, [selectedDisease])

  const updateField = (name, value) => {
    setFormState((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!selectedDisease) {
      return
    }

    setError(null)
    setResult(null)
    setLoading(true)

    const payload = {
      inputs: selectedDisease.fields.reduce((acc, field) => {
        const rawValue = formState[field.name]
        acc[field.name] = field.type === 'number' ? Number(rawValue) : rawValue
        return acc
      }, {}),
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict/${selectedDisease.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed')
      }
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (metadataError) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 px-4 py-8">
        <div className="mx-auto max-w-4xl rounded-3xl border border-red-500/30 bg-slate-900/90 p-10 text-center">
          <h1 className="text-3xl font-semibold text-red-300">Unable to load diagnosis models</h1>
          <p className="mt-4 text-slate-300">{metadataError}</p>
        </div>
      </div>
    )
  }

  if (!selectedDisease) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 px-4 py-8">
        <div className="mx-auto max-w-4xl rounded-3xl border border-slate-700 bg-slate-900/90 p-10 text-center">
          <p className="text-xl text-slate-300">Loading diagnosis models...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-slate-100 px-4 py-8">
      <div className="mx-auto max-w-7xl">
        <header className="mb-8 flex flex-col gap-4 rounded-3xl border border-slate-700 bg-slate-900/80 p-6 shadow-glow backdrop-blur-lg">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">AI Medical Diagnosis</p>
              <h1 className="mt-2 text-4xl font-semibold text-white">Interactive Health Prediction Dashboard</h1>
              <p className="mt-3 max-w-2xl text-slate-300">Choose a condition and enter values to compare them with the normal clinical range for that diagnosis.</p>
            </div>
            <div className="rounded-3xl bg-slate-950/70 p-4 text-slate-200 ring-1 ring-slate-700">
              <p className="text-sm uppercase tracking-[0.35em] text-cyan-200">API Status</p>
              <p className="mt-2 text-lg font-medium">Local backend at <span className="font-semibold text-cyan-300">/api</span></p>
            </div>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          <nav className="space-y-3 rounded-3xl border border-slate-700 bg-slate-900/75 p-4 shadow-lg shadow-cyan-500/10">
            <h2 className="mb-4 text-xl font-semibold text-white">Choose a diagnosis</h2>
            {diseases.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setSelectedId(item.id)}
                className={`w-full rounded-2xl border px-4 py-3 text-left transition ${
                  selectedId === item.id
                    ? 'border-cyan-400 bg-cyan-500/10 text-cyan-100 shadow-lg shadow-cyan-500/20'
                    : 'border-slate-700 bg-slate-950/80 text-slate-300 hover:border-slate-500 hover:bg-slate-900'
                }`}
              >
                <span className="block text-base font-semibold">{item.name}</span>
                <span className="mt-1 block text-sm text-slate-400">{item.description}</span>
              </button>
            ))}
          </nav>

          <main className="rounded-3xl border border-slate-700 bg-slate-900/80 p-6 shadow-lg shadow-slate-950/20">
            <div className="mb-6 flex flex-col gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.35em] text-cyan-300">{selectedDisease.name}</p>
                <h2 className="mt-2 text-3xl font-semibold text-white">Fill required inputs</h2>
              </div>
              <div className="rounded-3xl border border-slate-700 bg-slate-950/80 p-4 text-slate-200 ring-1 ring-slate-700">
                <p className="text-sm uppercase tracking-[0.35em] text-cyan-200">Normal range guidance</p>
                <p className="mt-3 text-sm text-slate-300">{getNormalRangeDetails(selectedDisease.id)}</p>
              </div>
              <div className="rounded-3xl bg-slate-950/80 px-4 py-3 text-slate-200 ring-1 ring-slate-700">
                <p className="text-sm">Fields</p>
                <p className="text-xl font-semibold text-white">{selectedDisease.fields.length}</p>
              </div>
            </div>

            <form className="grid gap-6" onSubmit={handleSubmit}>
              <div className="grid gap-4 sm:grid-cols-2">
                {selectedDisease.fields.map((field) => (
                  <label key={field.name} className="flex flex-col gap-2 rounded-3xl border border-slate-700 bg-slate-950/80 p-4">
                    <span className="text-sm font-medium text-slate-300">{field.label}</span>
                    {field.type === 'select' ? (
                      <select
                        value={formState[field.name]}
                        onChange={(e) => updateField(field.name, Number(e.target.value))}
                        className="rounded-2xl border border-slate-700 bg-slate-900 px-3 py-3 text-slate-100 transition hover:border-cyan-400 focus:border-cyan-400"
                      >
                        {field.options?.map((option) => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        value={formState[field.name]}
                        onChange={(e) => updateField(field.name, e.target.value)}
                        placeholder="0"
                        className="rounded-2xl border border-slate-700 bg-slate-950 px-3 py-3 text-slate-100 placeholder:text-slate-500 focus:border-cyan-400"
                      />
                    )}
                  </label>
                ))}
              </div>

              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center justify-center rounded-3xl bg-cyan-500 px-6 py-3 text-base font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? 'Running prediction...' : `Predict ${selectedDisease.name}`}
                </button>
              </div>
            </form>

            {error ? (
              <div className="mt-6 rounded-3xl border border-red-500/30 bg-red-500/10 p-5 text-red-100">
                <p className="font-semibold">Error</p>
                <p>{error}</p>
              </div>
            ) : null}

            {result ? (
              <div className="mt-6 rounded-3xl border border-cyan-500/20 bg-cyan-500/10 p-6 text-slate-100 backdrop-blur-sm">
                <p className="text-sm uppercase tracking-[0.35em] text-cyan-200">Prediction result</p>
                <h3 className="mt-3 text-3xl font-semibold text-white">{normalizeYesNoLabel(result.label)}</h3>
                <p className="mt-2 text-sm text-slate-300">Model: {result.model}</p>
                <p className="mt-3 rounded-3xl bg-slate-950/80 p-4 text-slate-300 ring-1 ring-slate-700">Details: {result.label}</p>
              </div>
            ) : null}
          </main>
        </div>
      </div>
    </div>
  )
}

export default App
