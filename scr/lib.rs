use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use rayon::prelude::*;
use ndarray::Array2;
use kmeans::{KMeans, Metric};

// =============== БАЗОВЫЕ СТРУКТУРЫ ===============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    pub id: String,
    pub object_type: String,
    pub properties: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Morphism {
    pub source: String,
    pub target: String,
    pub morphism_type: String,
    pub strength: f64,
    pub evidence: Vec<String>,
    pub timestamp: Option<f64>, // для временных данных
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Pattern {
    pub pattern_type: String,
    pub objects: Vec<String>,
    pub confidence: f64,
}

// =============== ФУНКТОРЫ ===============
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctorDirection {
    Lift,
    Project,
    Endo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Functor {
    pub name: String,
    pub source_level: u32,
    pub target_level: u32,
    pub direction: FunctorDirection,
    pub object_map: HashMap<String, String>,
}

impl Functor {
    pub fn is_endofunctor(&self) -> bool {
        self.source_level == self.target_level
    }
}

// =============== ЕСТЕСТВЕННЫЕ ПРЕОБРАЗОВАНИЯ ===============
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalTransformation {
    pub name: String,
    pub source_functor: String,
    pub target_functor: String,
    pub components: HashMap<String, Morphism>,
}

impl NaturalTransformation {
    pub fn is_natural(
        &self,
        src_category: &Category,
        tgt_category: &Category,
        functors: &HashMap<String, Functor>,
    ) -> bool {
        let f = functors.get(&self.source_functor);
        let g = functors.get(&self.target_functor);
        if f.is_none() || g.is_none() {
            return false;
        }
        let f = f.unwrap();
        let g = g.unwrap();

        for (x, eta_x) in &self.components {
            let fx = match f.object_map.get(x) {
                Some(id) => id,
                None => return false,
            };
            let gx = match g.object_map.get(x) {
                Some(id) => id,
                None => return false,
            };
            if eta_x.source != *fx || eta_x.target != *gx {
                return false;
            }

            for morph in &src_category.morphisms {
                if morph.source != *x {
                    continue;
                }
                let y = &morph.target;

                let fy = match f.object_map.get(y) {
                    Some(id) => id,
                    None => continue,
                };
                let ff = tgt_category.morphisms.iter().find(|m| &m.source == fx && &m.target == fy);

                let gy = match g.object_map.get(y) {
                    Some(id) => id,
                    None => continue,
                };
                let gf = tgt_category.morphisms.iter().find(|m| &m.source == gx && &m.target == gy);

                let eta_y = self.components.get(y);

                if let (Some(ff), Some(gf), Some(eta_y)) = (ff, gf, eta_y) {
                    let left = gf.strength * eta_x.strength;
                    let right = eta_y.strength * ff.strength;
                    if (left - right).abs() > 0.1 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// =============== МОНАДЫ (для процессов) ===============
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticMorphism {
    pub source: String,
    pub target: String,
    pub morphism_type: String,
    pub probability: f64,
    pub evidence: Vec<String>,
    pub timestamp: Option<f64>,
}

#[derive(Debug)]
pub struct ProcessCategory {
    pub name: String,
    pub objects: HashMap<String, std::sync::Arc<Object>>,
    pub processes: Vec<StochasticMorphism>,
}

impl ProcessCategory {
    pub fn new(name: String) -> Self {
        Self {
            name,
            objects: HashMap::new(),
            processes: Vec::new(),
        }
    }

    pub fn compose(&self, p1: &StochasticMorphism, p2: &StochasticMorphism) -> Option<StochasticMorphism> {
        if p1.target == p2.source {
            Some(StochasticMorphism {
                source: p1.source.clone(),
                target: p2.target.clone(),
                morphism_type: format!("{};{}", p1.morphism_type, p2.morphism_type),
                probability: p1.probability * p2.probability,
                evidence: vec!["composed".to_string()],
                timestamp: p1.timestamp.or(p2.timestamp),
            })
        } else {
            None
        }
    }

    pub fn find_two_step_processes(&self) -> Vec<StochasticMorphism> {
        let mut result = Vec::new();
        for p1 in &self.processes {
            for p2 in &self.processes {
                if let Some(composed) = self.compose(p1, p2) {
                    result.push(composed);
                }
            }
        }
        result
    }
}

// =============== ОСНОВНАЯ КАТЕГОРИЯ ===============
#[derive(Debug, Serialize, Deserialize)]
pub struct Category {
    pub name: String,
    pub objects: HashMap<String, std::sync::Arc<Object>>,
    pub morphisms: Vec<Morphism>,
}

impl Category {
    pub fn new(name: String) -> Self {
        Self {
            name,
            objects: HashMap::new(),
            morphisms: Vec::new(),
        }
    }

    pub fn add_objects_batch(&mut self, objects: Vec<Object>) {
        for obj in objects {
            self.objects.insert(obj.id.clone(), std::sync::Arc::new(obj));
        }
    }

    pub fn add_morphisms_batch(&mut self, morphisms: Vec<Morphism>) {
        self.morphisms.extend(morphisms);
    }

    pub fn clear(&mut self) {
        self.objects.clear();
        self.morphisms.clear();
    }

    pub fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("objects".to_string(), self.objects.len());
        stats.insert("morphisms".to_string(), self.morphisms.len());
        stats
    }

    pub fn get_morphisms(&self) -> Vec<HashMap<String, serde_json::Value>> {
        let mut result = Vec::new();
        for morph in &self.morphisms {
            let mut map = HashMap::new();
            map.insert("source".to_string(), serde_json::Value::String(morph.source.clone()));
            map.insert("target".to_string(), serde_json::Value::String(morph.target.clone()));
            map.insert("type".to_string(), serde_json::Value::String(morph.morphism_type.clone()));
            map.insert("strength".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(morph.strength).unwrap()));
            map.insert("timestamp".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(morph.timestamp.unwrap_or(0.0)).unwrap()));
            result.push(map);
        }
        result
    }

    pub fn get_object_ids(&self) -> Vec<String> {
        self.objects.keys().cloned().collect()
    }
}

// =============== ИЕРАРХИЯ ===============
#[derive(Debug)]
pub struct Hierarchy {
    categories: HashMap<u32, Category>,
    process_categories: HashMap<u32, ProcessCategory>,
    functors: HashMap<String, Functor>,
    natural_transformations: HashMap<String, NaturalTransformation>,
}

impl Hierarchy {
    pub fn new() -> Self {
        Self {
            categories: HashMap::new(),
            process_categories: HashMap::new(),
            functors: HashMap::new(),
            natural_transformations: HashMap::new(),
        }
    }

    pub fn add_category(&mut self, level: u32, category: Category) {
        self.categories.insert(level, category);
    }

    pub fn build_feature_matrix(&self, level: u32) -> Option<(Vec<String>, Array2<f64>)> {
        let cat = self.categories.get(&level)?;
        if cat.objects.is_empty() {
            return None;
        }

        let mut all_props = std::collections::HashSet::new();
        for obj in cat.objects.values() {
            for key in obj.properties.keys() {
                all_props.insert(key.clone());
            }
        }
        let all_props: Vec<String> = all_props.into_iter().collect();
        let n_props = all_props.len();
        let n_objects = cat.objects.len();

        let mut ids = Vec::with_capacity(n_objects);
        let mut matrix = Array2::zeros((n_objects, n_props));

        for (i, (id, obj)) in cat.objects.iter().enumerate() {
            ids.push(id.clone());
            for (j, prop) in all_props.iter().enumerate() {
                matrix[[i, j]] = *obj.properties.get(prop).unwrap_or(&0.0);
            }
        }

        Some((ids, matrix))
    }

    pub fn auto_lift(&mut self, from_level: u32, to_level: u32, n_clusters: usize) -> Result<(), String> {
        if from_level >= to_level || n_clusters == 0 {
            return Err("Invalid levels or clusters".to_string());
        }

        let (ids, matrix) = self.build_feature_matrix(from_level)
            .ok_or("No objects to lift".to_string())?;

        if n_clusters >= ids.len() {
            return Err("Too many clusters".to_string());
        }

        let kmeans = KMeans::new(n_clusters, 100, Metric::Euclidean);
        let (_centroids, assignments) = kmeans.fit(&matrix).map_err(|_| "Clustering failed".to_string())?;

        let mut new_cat = Category::new(format!("level_{}", to_level));
        let mut object_map = HashMap::new();

        for (i, &cluster_id) in assignments.iter().enumerate() {
            let concrete_id = &ids[i];
            let abstract_id = format!("L{}_C{}", to_level, cluster_id);
            object_map.insert(concrete_id.clone(), abstract_id);
        }

        let src_cat = self.categories.get(&from_level).unwrap();
        for (src_id, tgt_id) in &object_map {
            if let Some(obj) = src_cat.objects.get(src_id) {
                new_cat.objects.insert(tgt_id.clone(), std::sync::Arc::new(Object {
                    id: tgt_id.clone(),
                    object_type: "abstract".to_string(),
                    properties: obj.properties.clone(),
                }));
            }
        }

        for morph in &src_cat.morphisms {
            if let (Some(src_tgt), Some(tgt_tgt)) = (
                object_map.get(&morph.source),
                object_map.get(&morph.target)
            ) {
                new_cat.morphisms.push(Morphism {
                    source: src_tgt.clone(),
                    target: tgt_tgt.clone(),
                    morphism_type: morph.morphism_type.clone(),
                    strength: morph.strength,
                    evidence: vec!["lifted".to_string()],
                    timestamp: morph.timestamp,
                });
            }
        }

        self.add_category(to_level, new_cat);
        let functor = Functor {
            name: format!("lift_{}_{}", from_level, to_level),
            source_level: from_level,
            target_level: to_level,
            direction: FunctorDirection::Lift,
            object_map,
        };
        self.functors.insert(functor.name.clone(), functor);

        Ok(())
    }

    pub fn build_temporal_endofunctor(&mut self, level: u32, start: f64, end: f64) -> Result<String, String> {
        let cat = self.categories.get_mut(&level)
            .ok_or("Category not found".to_string())?;

        let mut object_map = HashMap::new();
        for id in cat.objects.keys() {
            object_map.insert(id.clone(), id.clone());
        }

        let functor_name = format!("endo_t{}_{}", start as i64, end as i64);
        let functor = Functor {
            name: functor_name.clone(),
            source_level: level,
            target_level: level,
            direction: FunctorDirection::Endo,
            object_map,
        };
        self.functors.insert(functor_name.clone(), functor);

        Ok(functor_name)
    }

    pub fn build_process_category(&mut self, level: u32) -> Result<String, String> {
        let cat = self.categories.get(&level)
            .ok_or("Category not found".to_string())?;

        let mut proc_cat = ProcessCategory::new(format!("process_{}", level));
        for morph in &cat.morphisms {
            proc_cat.processes.push(StochasticMorphism {
                source: morph.source.clone(),
                target: morph.target.clone(),
                morphism_type: morph.morphism_type.clone(),
                probability: morph.strength, // сила как вероятность
                evidence: morph.evidence.clone(),
                timestamp: morph.timestamp,
            });
        }

        self.process_categories.insert(level + 2000, proc_cat);
        Ok(format!("process_{}", level))
    }

    pub fn add_natural_transformation(&mut self, nt: NaturalTransformation) {
        self.natural_transformations.insert(nt.name.clone(), nt);
    }

    pub fn compare_functors(&self, f1_name: &str, f2_name: &str, level: u32) -> Result<NaturalTransformation, String> {
        let cat = self.categories.get(&level)
            .ok_or("Category not found".to_string())?;
        let f1 = self.functors.get(f1_name)
            .ok_or("Functor 1 not found".to_string())?;
        let f2 = self.functors.get(f2_name)
            .ok_or("Functor 2 not found".to_string())?;

        let mut components = HashMap::new();
        for (x, _) in &cat.objects {
            if let (Some(fx1), Some(fx2)) = (f1.object_map.get(x), f2.object_map.get(x)) {
                components.insert(x.clone(), Morphism {
                    source: fx1.clone(),
                    target: fx2.clone(),
                    morphism_type: "natural_component".to_string(),
                    strength: 0.5,
                    evidence: vec!["auto".to_string()],
                    timestamp: None,
                });
            }
        }

        Ok(NaturalTransformation {
            name: format!("{}->{}", f1_name, f2_name),
            source_functor: f1_name.to_string(),
            target_functor: f2_name.to_string(),
            components,
        })
    }

    // Методы для PyO3
    pub fn get_category_stats(&self, level: u32) -> Option<HashMap<String, usize>> {
        self.categories.get(&level).map(|c| c.stats())
    }

    pub fn get_category_morphisms(&self, level: u32) -> Option<Vec<HashMap<String, serde_json::Value>>> {
        self.categories.get(&level).map(|c| c.get_morphisms())
    }

    pub fn get_category_object_ids(&self, level: u32) -> Option<Vec<String>> {
        self.categories.get(&level).map(|c| c.get_object_ids())
    }

    pub fn get_process_two_step(&self, level: u32) -> Option<Vec<HashMap<String, serde_json::Value>>> {
        self.process_categories.get(&level).map(|pc| {
            pc.find_two_step_processes().into_iter().map(|p| {
                let mut map = HashMap::new();
                map.insert("source".to_string(), serde_json::Value::String(p.source));
                map.insert("target".to_string(), serde_json::Value::String(p.target));
                map.insert("morphism_type".to_string(), serde_json::Value::String(p.morphism_type));
                map.insert("probability".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(p.probability).unwrap()));
                map
            }).collect()
        })
    }

    pub fn get_levels(&self) -> Vec<u32> {
        let mut levels: Vec<u32> = self.categories.keys().cloned().collect();
        levels.sort();
        levels
    }

    pub fn get_functor_names(&self) -> Vec<String> {
        self.functors.keys().cloned().collect()
    }
}

// =============== PyO3 ОБЁРТКИ ===============

#[pyclass]
pub struct PyCategory {
    inner: Category,
}

#[pymethods]
impl PyCategory {
    #[new]
    fn new(name: String) -> Self {
        Self { inner: Category::new(name) }
    }

    fn add_objects_batch(&mut self, objects: Vec<PyObject>) -> PyResult<()> {
        let rust_objects: Result<Vec<Object>, PyErr> = objects
            .into_iter()
            .map(|obj_py| {
                let dict = obj_py.cast_as::<PyDict>(pyo3::Python::acquire_gil().python())?;
                let id = dict.get_item("id")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("'id' is required"))?
                    .extract::<String>()?;
                let object_type = dict.get_item("type")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("'type' is required"))?
                    .extract::<String>()?;
                let properties = dict.get_item("properties")
                    .map(|p| p.extract::<HashMap<String, f64>>())
                    .transpose()?
                    .unwrap_or_default();
                Ok(Object { id, object_type, properties })
            })
            .collect();
        self.inner.add_objects_batch(rust_objects?);
        Ok(())
    }

    fn add_morphisms_batch(&mut self, morphisms: Vec<PyObject>) -> PyResult<()> {
        let rust_morphisms: Result<Vec<Morphism>, PyErr> = morphisms
            .into_iter()
            .map(|morph_py| {
                let dict = morph_py.cast_as::<PyDict>(pyo3::Python::acquire_gil().python())?;
                let source = dict.get_item("source")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("'source' is required"))?
                    .extract::<String>()?;
                let target = dict.get_item("target")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("'target' is required"))?
                    .extract::<String>()?;
                let morphism_type = dict.get_item("type")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("'type' is required"))?
                    .extract::<String>()?;
                let strength = dict.get_item("strength")
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("'strength' is required"))?
                    .extract::<f64>()?;
                let timestamp = dict.get_item("timestamp")
                    .and_then(|t| t.extract::<f64>().ok());
                let evidence = dict.get_item("evidence")
                    .map(|e| e.extract::<Vec<String>>())
                    .transpose()?
                    .unwrap_or_default();
                Ok(Morphism { source, target, morphism_type, strength, evidence, timestamp })
            })
            .collect();
        self.inner.add_morphisms_batch(rust_morphisms?);
        Ok(())
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn stats(&self) -> PyResult<HashMap<String, usize>> {
        Ok(self.inner.stats())
    }

    fn get_morphisms(&self) -> PyResult<Vec<HashMap<String, serde_json::Value>>> {
        Ok(self.inner.get_morphisms())
    }

    fn get_object_ids(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.get_object_ids())
    }
}

#[pyclass]
pub struct PyHierarchy {
    inner: Hierarchy,
}

#[pymethods]
impl PyHierarchy {
    #[new]
    fn new() -> Self {
        Self { inner: Hierarchy::new() }
    }

    fn set_base_category(&mut self, cat: &PyCategory) -> PyResult<()> {
        self.inner.add_category(0, cat.inner.clone());
        Ok(())
    }

    fn auto_lift(&mut self, from_level: u32, to_level: u32, n_clusters: usize) -> PyResult<()> {
        self.inner.auto_lift(from_level, to_level, n_clusters)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    fn build_temporal_endofunctor(&mut self, level: u32, start: f64, end: f64) -> PyResult<String> {
        self.inner.build_temporal_endofunctor(level, start, end)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    fn build_process_category(&mut self, level: u32) -> PyResult<String> {
        self.inner.build_process_category(level)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    fn compare_functors(&self, f1: String, f2: String, level: u32) -> PyResult<PyNaturalTransformation> {
        let nt = self.inner.compare_functors(&f1, &f2, level)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        Ok(PyNaturalTransformation { inner: nt })
    }

    fn get_category_stats(&self, level: u32) -> PyResult<HashMap<String, usize>> {
        self.inner.get_category_stats(level)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Category not found"))
    }

    fn get_category_morphisms(&self, level: u32) -> PyResult<Vec<HashMap<String, serde_json::Value>>> {
        self.inner.get_category_morphisms(level)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Category not found"))
    }

    fn get_category_object_ids(&self, level: u32) -> PyResult<Vec<String>> {
        self.inner.get_category_object_ids(level)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Category not found"))
    }

    fn get_process_two_step(&self, level: u32) -> PyResult<Vec<HashMap<String, serde_json::Value>>> {
        self.inner.get_process_two_step(level)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Process category not found"))
    }

    fn get_levels(&self) -> PyResult<Vec<u32>> {
        Ok(self.inner.get_levels())
    }

    fn get_functor_names(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.get_functor_names())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyNaturalTransformation {
    inner: NaturalTransformation,
}

#[pymethods]
impl PyNaturalTransformation {
    #[new]
    fn new(name: String, source_functor: String, target_functor: String) -> Self {
        Self {
            inner: NaturalTransformation {
                name,
                source_functor,
                target_functor,
                components: HashMap::new(),
            }
        }
    }

    fn add_component(&mut self, object_id: String, source: String, target: String, strength: f64) {
        self.inner.components.insert(object_id, Morphism {
            source,
            target,
            morphism_type: "natural_component".to_string(),
            strength,
            evidence: vec![],
            timestamp: None,
        });
    }

    fn is_natural(&self, hierarchy: &PyHierarchy) -> PyResult<bool> {
        // Упрощённо: берём категории 0 и 1
        let src_cat = hierarchy.inner.categories.get(&0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Source category not found"))?;
        let tgt_cat = hierarchy.inner.categories.get(&1)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Target category not found"))?;

        Ok(self.inner.is_natural(src_cat, tgt_cat, &hierarchy.inner.functors))
    }

    fn get_components(&self) -> PyResult<Vec<(String, HashMap<String, serde_json::Value>)>> {
        let result: Vec<_> = self.inner.components.iter().map(|(k, v)| {
            let mut map = HashMap::new();
            map.insert("source".to_string(), serde_json::Value::String(v.source.clone()));
            map.insert("target".to_string(), serde_json::Value::String(v.target.clone()));
            map.insert("strength".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(v.strength).unwrap()));
            (k.clone(), map)
        }).collect();
        Ok(result)
    }
}

// =============== ЭКСПОРТ МОДУЛЯ ===============
#[pymodule]
fn categorical_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCategory>()?;
    m.add_class::<PyHierarchy>()?;
    m.add_class::<PyNaturalTransformation>()?;
    Ok(())
}
