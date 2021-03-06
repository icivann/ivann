<template>
  <span @click="deleteEditor">
    <i class="delete-button fa fa-trash"/>
  </span>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import EditorType from '@/EditorType';
import { Getter, Mutation } from 'vuex-class';
import { saveEditor, SaveWithNames } from '@/file/EditorAsJson';
import { EditorModel } from '@/store/editors/types';

@Component
export default class DeleteEditorButton extends Vue {
  @Prop({ required: true }) readonly editorType!: EditorType;
  @Prop({ required: true }) readonly index!: number;
  @Prop({ required: true }) readonly name!: string;
  @Getter('overviewEditor') overviewEditor!: EditorModel;
  @Getter('saveWithNames') saveWithNames!: SaveWithNames;
  @Mutation('deleteEditor') delete!: (arg0: { editorType: EditorType; index: number }) => void;

  private deleteEditor() {
    if (window.confirm('Are you sure you want to delete this editor? '
      + 'All unsaved progress will be lost.')) {
      this.delete({ editorType: this.editorType, index: this.index });

      // Re-save overview after nodes may have been removed
      const overviewEditorSave = saveEditor(this.overviewEditor);
      localStorage.setItem('unsaved-editor-Overview', JSON.stringify(overviewEditorSave));

      // Remove deleted editor from Local Storage.
      localStorage.removeItem(`unsaved-editor-${this.name}`);
      localStorage.setItem('unsaved-project', JSON.stringify(this.saveWithNames));
    }
  }
}
</script>

<style lang="scss" scoped>
  .delete-button {
    color: var(--foreground);
    background: none;
    padding: 0 4px 1px;
    border-radius: 2px;
    margin: 0 4px;
    &:hover {
      background: var(--red);
    }
  }
</style>
