<template>
  <div>
    Export the selected elements
    <CheckboxField :checked="overviewSelected" label="Overview"
                   @value-change="overviewSelected = $event"/>
    <CheckboxList label="Models" :children="modelEditors.map((child) => child.name)"
                  v-show="modelEditors.length > 0" v-model="selectedModels"/>
    <CheckboxList label="Datasets" :children="dataEditors.map((child) => child.name)"
                  v-show="dataEditors.length > 0" v-model="selectedData"/>
    <CheckboxList label="Codevault" :children="files.map((child) => child.filename)"
                  v-show="files.length > 0" v-model="selectedFiles"/>
    <div class="buttons">
      <UIButton text="Cancel" @click="closeModal"/>
      <UIButton text="Export" :primary="true" @click="exportCode"/>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import UIButton from '@/components/buttons/UIButton.vue';
import CheckboxList from '@/components/CheckboxList.vue';
import { mapGetters } from 'vuex';
import { downloadZip } from '@/file/Utils';
import { saveEditor } from '@/file/EditorAsJson';
import istateToGraph from '@/app/ir/istateToGraph';
import EditorType from '@/EditorType';
import { generateModelCode, generateOverviewCode } from '@/app/codegen/codeGenerator';
import Graph from '@/app/ir/Graph';
import Modal from '@/components/modals/Modal.vue';
import { Getter } from 'vuex-class';
import { EditorModel, EditorModels } from '@/store/editors/types';
import CheckboxValue from '@/baklava/CheckboxValue';
import CheckboxField from '@/components/CheckboxField.vue';
import JSZip from 'jszip';
import generateData from '@/app/codegen/dataGenerator';
import { ParsedFile } from '@/store/codeVault/types';

@Component({
  components: {
    CheckboxField,
    CheckboxList,
    UIButton,
    Modal,
  },
  computed: mapGetters(['modelEditors', 'dataEditors', 'files']),
})
export default class ExportModal extends Vue {
  @Getter('currEditorType') currEditorType!: EditorType;
  @Getter('allEditorModels') editorModels!: EditorModels;
  @Getter('currEditorModel') currEditor!: EditorModel;

  private selectedModels: Array<string> = [];
  private selectedData: Array<string> = [];
  private selectedFiles: Array<string> = [];
  private overviewSelected: CheckboxValue = CheckboxValue.CHECKED;

  private closeModal() {
    this.$parent.$emit('input', false);
  }

  private exportCode() {
    console.log(`Models to be exported: ${this.selectedModels}`);
    console.log(`Data to be exported: ${this.selectedData}`);
    console.log(`Files to be exported: ${this.selectedFiles}`);
    console.log(`Should export Overview?: ${this.overviewSelected === CheckboxValue.CHECKED}`);

    const zip = new JSZip();

    const modelsFolder = zip.folder('models');
    const dataFolder = zip.folder('data');
    const codevaultFolder = zip.folder('codevault');

    // generate overview
    if (this.overviewSelected) {
      const { name, state } = saveEditor(this.currEditor);
      const graph = istateToGraph(state);
      console.log('generating overview');
      const models = this.editorModels.modelEditors.map((editor) => {
        const { name, state } = saveEditor(editor);
        const graph = istateToGraph(state);
        return [graph, name] as [Graph, string];
      });
      const data = this.editorModels.dataEditors.map((editor) => {
        const { name, state } = saveEditor(editor);
        const graph = istateToGraph(state);
        return [graph, name] as [Graph, string];
      });
      const contents = generateOverviewCode(graph, models, data);
      zip.file('main.py', contents);
    }
    // generate model files
    if (modelsFolder !== null) {
      this.editorModels.modelEditors.filter((editor) => this.selectedModels.includes(editor.name))
        .forEach(
          (editor) => {
            const { name, state } = saveEditor(editor);
            const graph = istateToGraph(state);
            const contents = generateModelCode(graph, name);
            modelsFolder.file(`${name}.py`, contents);
          },
        );
    }
    // generate data files
    if (dataFolder !== null) {
      this.editorModels.dataEditors
        .filter((editor) => this.selectedData.includes(editor.name))
        .forEach(
          (editor) => {
            const { name, state } = saveEditor(editor);
            const graph = istateToGraph(state);
            const contents = generateData(graph, name);
            dataFolder.file(`${name}.py`, contents);
          },
        );
    }
    // generate codevault files
    if (codevaultFolder !== null) {
      (this.$store.state.codeVault.files as ParsedFile[])
        .filter((file) => this.selectedFiles.includes(file.filename))
        .forEach(
          (file) => {
            codevaultFolder.file(file.filename, file.functions.join('\n'));
          },
        );
    }

    console.log('invoking download python');
    downloadZip('ivann', zip);

    this.closeModal();
  }
}
</script>

<style scoped>
.buttons {
  display: flex;
  float: right;
  margin-top: 0.75rem;
}

.button {
  margin-left: 1rem;
}
</style>
