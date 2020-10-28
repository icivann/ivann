<template>
  <div>
    <ExpandablePanel name="Models">
      <ButtonGrid>
        <AddNodeButton v-for="editor in editorList"
                       node="ModelEncapsulation"
                       :options="editor"
                       :key="editor.name"
                       :name="editor.name"/>
      </ButtonGrid>
    </ExpandablePanel>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import { EditorModel } from '@/store/editors/types';
import ModelEncapsulation, { ModelOption } from '@/nodes/overview/ModelEncapsulation';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
})
export default class ComponentsTab extends Vue {
  private editorList: ModelOption[] = [];

  private created() {
    const { currEditorModel, modelEditors } = this.$store.getters;
    const { editor } = (currEditorModel as EditorModel);
    for (const node of editor.nodes) {
      // TODO replace the following with UUIDs once implemented
      for (const modelEditor of modelEditors as EditorModel[]) {
        if (modelEditor.name === node.name) {
          (node as ModelEncapsulation).update(modelEditor);
        }
      }
    }
    this.makeOptions(this.$store.getters.modelEditors);
  }

  @Watch('$store.getters.modelEditors')
  private onEditorsUpdate(editors: EditorModel[]): void {
    this.makeOptions(editors);
  }

  private makeOptions(editors: EditorModel[]): void {
    this.editorList = editors.map((editor) => ({
      name: editor.name,
      inputs: editor.inputs,
      outputs: editor.outputs,
    }));
  }
}
</script>
